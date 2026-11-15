
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from statsmodels.tsa.stattools import adfuller

from scipy.signal import welch
from scipy.stats import kurtosis, skew
import lightgbm as lgb



# Pre-define constants outside the loop to avoid re-allocation
FS = 20480  # Sampling rate [Hz]
NPERSEG = 128  # Welch window length
BANDS = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 512)]


def _extract_single_file_features(args):
  """Worker function to extract features for a single file (runs in parallel)."""
  file_idx, sequences = args
  C, L = sequences.shape
  feats = {}

  # Pre-compute Welch once per file if possible, or per channel
  # (Welch on 20480 points with nperseg=128 is fast, but we can optimize band extraction)
  
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
      adf_p = adfuller(x, maxlag=1, autolag=None).pvalue
    except Exception:
      # assumption
      adf_p = 1.0

    # ----- Frequency-domain (Welch) -----
    f, Pxx = welch(x, fs=FS, nperseg=NPERSEG)

    # Vectorized band power calculation using vectorized NumPy search
    band_powers = np.empty(len(BANDS))
    for b_idx, (lo, hi) in enumerate(BANDS):
      idx = (f >= lo) & (f < hi)
      band_powers[b_idx] = np.trapezoid(Pxx[idx], f[idx]) if np.any(idx) else 0.0

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

  return file_idx, feats


def extract_features_parallel(all_seqs):
  n_files = all_seqs.shape[0]
  feature_list = [None] * n_files

  tasks = [(i, all_seqs[i]) for i in range(n_files)]

  print(f'Extracting features across {n_files} files in parallel...')
  with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(_extract_single_file_features, task) for task in tasks
    ]

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



# def chunk_dataset(data, chunk_len=1024, overlap=0.5):
#     files, channels, length = data.shape

#     # 1. Non-overlapping chunks (safely truncates any remainder)
#     if overlap <= 0:
#       tot_chunks = length // chunk_len
#       trimmed = data[:, :, :tot_chunks * chunk_len]
#       return trimmed.reshape(files, channels, tot_chunks, chunk_len), tot_chunks

#     # 2. Overlapping chunks using explicit stride = chunk_len * (1 - overlap)
#     if overlap < 1:
#       stride = max(1, int(chunk_len * (1 - overlap)))
#       chunks = sliding_window_view(data, window_shape=chunk_len, axis=2)[:, :, ::stride, :]
#       tot_chunks = len(chunks)
#       return chunks, tot_chunks


# def standardize(data:torch.Tensor, mean:torch.Tensor=None, std:torch.Tensor=None):
#   if mean is None:
#       mean = torch.mean(data, dim=(0,2),keepdim=True)
#   if std is None:
#       std = torch.std(data, dim=(0,2),keepdim=True)
#   return (data - mean) / (std + 1e-9), (mean, std)

# def destandardize(data:torch.Tensor, mean:torch.Tensor, std:torch.Tensor):
#   mean = mean.view(1, -1, 1)
#   std = std.view(1, -1, 1)
#   return data * (std + 1e-9) + mean

# # def create_sensor_dataset(dfs:List[pd.DataFrame], aggregate_labels:bool=True, seq_length_seconds:float=1., do_sampling_rate_estimation:bool=False, do_standardize:bool=False, standardize_params:tuple=None, do_detrend:bool=False, detrend_method:str='mean', detrend_mean:torch.Tensor=None) -> SensorDataset:
# def prepare_dloader(data, standardize_params:tuple=None, do_standardize:bool=False,batch_size:int=32, shuffle:bool=True, is_test=False):
#   # 1. transform to torch
#   # 2. reshape to (samples, channels, length)
#   # 3. standardize
#   # 4. create DataSet and DataLoader

#   files,channels,chunks,length = data.shape
#   data = torch.as_tensor(data).float().permute(0, 2, 1, 3).contiguous()
#   data = data.reshape(-1, channels, length)

#   if do_standardize:
#     # standardize the data
#     mean, std = None, None
#     if standardize_params is not None and len(standardize_params) == 2:
#       mean, std = standardize_params
#     data, (mean, std) = standardize(data, mean, std)

#   if is_test:
#     labels = torch.empty(files*chunks)
#   else:
#     labels = torch.zeros(files*chunks)
#   dataset = TensorDataset(data, labels)


#   loader = DataLoader(dataset, shuffle=shuffle, batch_size=32)

#   params = {}
#   if do_standardize:
#       params['standardize'] = {'mean': mean, 'std': std}
#   return loader, params

# def restore_dataset(predictions: torch.Tensor,n_files: int,n_chunks: int,original_length: int = None):
    """
    Restore flattened chunk predictions to original recordings.

    Input:
        (files * chunks_per_file, channels, chunk_len)

    Output:
        (files, channels, original_length)
    """

    total_chunks, channels, chunk_len = predictions.shape

    assert total_chunks == n_files * n_chunks, (
        f"Expected {n_files * n_chunks} chunks, "
        f"received {total_chunks}"
    )

    # Reverse reshape performed in prepare_dloader
    restored = predictions.reshape(n_files, n_chunks, channels, chunk_len)

    # Reverse permutation performed in prepare_dloader
    restored = restored.permute(0, 2, 1, 3).contiguous()

    # Concatenate temporal chunks
    restored = restored.reshape(n_files, channels, n_chunks * chunk_len)

    # Optionally recover original sequence length
    if original_length is not None:
        restored = restored[:, :, :original_length]

    return restored