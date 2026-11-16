# Anomaly Detection su dati di vibrazione di cuscinetti
*Autore: **Gregorio Piqué** • Data: 17/11/2026*
<!-- # Bearing Anomaly Detection Service -->

Servizio REST basato su FastAPI e Docker per il rilevamento di anomalie su dati di vibrazione di cuscinetti (NASA IMS Bearing Dataset - Set 2).

## Struttura del Repository

```text
├── app/
│   └── main.py                         # Endpoint FastAPI (/, /train, /predict)
├── data/
│   └── data_utils.py                   # Caricamento e preprocessing dei segnali
├── docs/
│   └── anomaly_detection_yanmar.pdf    # Report del progetto
├── models/
│   ├── FeatEngOCSVM.py                 # Classe modello (feature-engineered-based One-Class SVM)
│   └── configs/                        # Modelli addestrati (.joblib)
├── notebooks/
│   └── anomaly_detection_yanmar.ipynb  # Esplorazione dati e test (Parte 1)
├── src/
│   └── pipeline.py                     # Pipeline di training e inferenza
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Avvio con Docker

1. **Build e avvio del container:**
   ```bash
   docker compose up --build
   ```
   Il servizio sarà esposto sulla porta `8000`.

2. **Accesso tramite browser:**
   * `http://localhost:8000/`: interfaccia web minimale per caricare direttamente i file sugli endpoint `/train` e `/predict`.
   * `http://localhost:8000/docs`: documentazione interattiva Swagger UI.

3. **Arresto:**
   ```bash
   docker compose down
   ```

## Utilizzo delle API

Entrambi gli endpoint accettano uno o più file di acquisizione grezzi tramite `multipart/form-data` (`List[UploadFile]`).

### Training (`POST /train`)
Addestra un nuovo modello sui file forniti e salva gli artefatti in `models/configs/`.

```bash
curl -X POST "http://localhost:8000/train" \
  -F "files=@/percorso/2nd_test/2004.02.12.10.32.39" \
  -F "files=@/percorso/2nd_test/2004.02.12.10.42.39"
```

### Inferenza (`POST /predict`)
Restituisce score ed labels di anomalia per ciascun canale sui file forniti.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "files=@/percorso/2nd_test/2004.02.18.00.02.39" \
  -F "files=@/percorso/2nd_test/2004.02.19.06.12.39"
```