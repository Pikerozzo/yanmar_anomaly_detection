
from http.client import HTTPException
import io
from typing import List
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastapi import UploadFile

from statsmodels.tsa.stattools import adfuller

from scipy.signal import welch
from scipy.stats import kurtosis, skew



# Pre-define constants outside the loop to avoid re-allocation
FS = 20480  # Sampling rate [Hz]
NPERSEG = 128  # Welch window length
BANDS = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 512)]


def _extract_single_file_features(args):
  """Worker function to extract features for a single file (runs in parallel)."""
  file_idx, sequences = args
  C, L = sequences.shape
  feats = {}

  for c in range(C):
    x = sequences[c].astype(np.float32)
    prefix = f'ch{c}_'

    # ----- Time-domain (Vectorized via NumPy/SciPy) -----
    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x**2))
    x_max, x_min = x.max(), x.min()
    p2p = x_max - x_min

    # Fast NumPy-native skew & kurtosis (replaces pd.Series overhead)
    sk = skew(x)
    kur = kurtosis(x)  # Fisher's definition (normal = 0) or Pearson's depending on version

    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    crest = x_max / rms if rms > 0 else 0.0
    pct_over_3std = np.mean(np.abs(x - mean) > 3 * std)

    # ----- Stationarity -----
    try:
      adf_p = float(adfuller(x, maxlag=1, autolag=None,result_object=True).pvalue)
    except:
      # assumption
      adf_p = 1.0

    # ----- Frequency-domain (Welch) -----
    f, Pxx = welch(x, fs=FS, nperseg=NPERSEG)

    # Vectorized band power calculation using vectorized NumPy search
    band_powers = np.empty(len(BANDS))
    for b_idx, (lo, hi) in enumerate(BANDS):
      idx = (f >= lo) & (f < hi)
      band_powers[b_idx] = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0

    low_power = band_powers[0]
    high_power = band_powers[-1]
    hi_lo_ratio = high_power / (low_power + 1e-12)

    # Two tallest peaks in PSD
    peak_idx = np.argsort(Pxx)[-2:][::-1]
    pk1_f, pk1_a = f[peak_idx[0]], Pxx[peak_idx[0]]
    pk2_f, pk2_a = f[peak_idx[1]], Pxx[peak_idx[1]]

    feats.update({
        prefix + 'mean': mean,
        prefix + 'std': std,
        prefix + 'rms': rms,
        prefix + 'max': x_max,
        prefix + 'min': x_min,
        prefix + 'peak_to_peak': p2p,
        prefix + 'skew': sk,
        prefix + 'kurtosis': kur,
        prefix + 'mad': mad,
        prefix + 'crest_factor': crest,
        prefix + 'pct_over_3std': pct_over_3std,
        prefix + 'adf_pvalue': adf_p,
        prefix + 'hi_lo_ratio': hi_lo_ratio,
        prefix + 'peak1_freq': pk1_f,
        prefix + 'peak1_amp': pk1_a,
        prefix + 'peak2_freq': pk2_f,
        prefix + 'peak2_amp': pk2_a,
    })

  # ----- Cross-channel features -----
  if C > 1:
    corr = np.corrcoef(sequences)
    upper = corr[np.triu_indices(C, k=1)]
    feats['cross_corr_mean'] = np.nanmean(upper)
    feats['cross_corr_std'] = np.nanstd(upper)

    var_per_ch = sequences.var(axis=1)
    feats['var_ratio_max_min'] = var_per_ch.max() / (var_per_ch.min() + 1e-12)

  feats = {k: float(v) for k, v in feats.items()}
  return file_idx, feats


def extract_features_parallel(all_seqs):
  n_files = all_seqs.shape[0]
  feature_list = [None] * n_files

  tasks = [(i, all_seqs[i]) for i in range(n_files)]

  print(f'Extracting features across {n_files} files in parallel...')
  with ProcessPoolExecutor() as executor:
    futures = [executor.submit(_extract_single_file_features, task) for task in tasks]

    for future in as_completed(futures):
      file_idx, feats = future.result()
      feature_list[file_idx] = feats
  return pd.DataFrame(feature_list)

def extract_features(data):
  features = extract_features_parallel(data)
  channel_feature_dfs = {}
  for ch in range(4):
    ch_name = f'ch{ch}'
    ch_cols = [col for col in features.columns if col.startswith(ch_name)]
    channel_feature_dfs[ch] = features[ch_cols].copy()
  return channel_feature_dfs



def _load_single_file_task(args):
  idx, file_bytes = args
  arr = (pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None, engine="c").to_numpy().T)
  return idx, arr


async def load_all_files_parallel(files: List[UploadFile]) -> np.ndarray:
  """Loads N uploaded files in parallel and packs them directly into a pre-allocated

  3D NumPy array of shape (N_files, 4_channels, M_measurements).
  """
  n_files = len(files)

  if n_files == 0:
    raise ValueError("No files found to load.")

  file_contents = []
  for f in files:
    content = await f.read()
    file_contents.append(content)
    await f.close()

  _, first_arr = _load_single_file_task((0, file_contents[0]))
  n_channels, n_measurements = first_arr.shape  # Expected: (4, 20480)

  data = np.empty((n_files, n_channels, n_measurements), dtype=np.float64)
  data[0] = first_arr

  tasks = [(i, content) for i, content in enumerate(file_contents[1:], start=1)]

  print(f"Loading {n_files} files in parallel...")
  with ProcessPoolExecutor() as executor:
    futures = [executor.submit(_load_single_file_task, task) for task in tasks]

    for future in as_completed(futures):
      idx, arr = future.result()
      data[idx] = arr

  print(f"Successfully loaded shape: {data.shape}")
  return data

async def load_and_extract_features(files):
  try:
    data = await load_all_files_parallel(files)
    data_feats = extract_features(data)
    data_feats = {
            f"ch{ch}": df.to_dict(orient="records")
            for ch, df in data_feats.items()
        }
    return data, data_feats
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Errore interno durante la predizione: {str(e)}")