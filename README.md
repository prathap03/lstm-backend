---
title: LstmChurn
emoji: 🌍
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# LSTM Churn Prediction API

A FastAPI service that serves a Keras/TensorFlow LSTM classifier for **telecom customer churn prediction**. It scores a single customer from a JSON payload, or a whole CSV of customers in one request, and returns the predicted class and the churn probability.

It is the backend of a churn-prediction web application (Next.js front end) built during a deep learning internship at SR University, and it is containerized with Docker so it can be hosted anywhere that runs containers. The block at the top of this file is Hugging Face Spaces configuration; keep it if you host the API on Spaces.

## Tech stack

- Python 3.9, FastAPI, Uvicorn
- Keras 2.11 / TensorFlow 2.11 (LSTM classifier)
- pandas, NumPy, scikit-learn
- Docker

## Model and input features

The trained model is in [`model/`](model/) (`lstmclass.h5` and `lstmclass.keras`). It takes **10 numeric features per customer**, in this order:

| # | Feature |
|---|---|
| 1 | `international_plan` (0/1) |
| 2 | `voice_mail_plan` (0/1) |
| 3 | `number_vmail_messages` |
| 4 | `total_intl_minutes` |
| 5 | `total_intl_calls` |
| 6 | `total_intl_charge` |
| 7 | `number_customer_service_calls` |
| 8 | `total_net_minutes` |
| 9 | `total_net_calls` |
| 10 | `total_net_charge` |

[`dataset/pre_train.csv`](dataset/pre_train.csv) is a sample of about 3,400 preprocessed customers in exactly this format, and a good test file for the CSV endpoint.

## Background: the churn model

The full project is a hybrid ensemble for churn prediction:

1. **Preprocessing** with pandas, NumPy and TensorFlow: duplicate and null removal, normalization, and a train/test split.
2. **Feature selection** with BAOA (Binary Arithmetic Optimization Algorithm).
3. **LSTM autoencoder**: an encoder block compresses the selected features into a latent space, a decoder block follows, and fully connected layers produce the churn / no-churn output.
4. **Voting ensemble** combining the LSTM with Logistic Regression, Random Forest and Decision Tree classifiers.

This repository serves the trained **LSTM model** (`model/lstmclass.h5` and `lstmclass.keras`) behind the API below. The other ensemble members and the training code are not part of this repo.

### Results

Evaluated on a held-out test set of 850 customers (721 retained, 129 churned):

| Metric | Value |
|---|---|
| Accuracy | 0.975 |
| ROC AUC | 0.94 |
| Churn class: precision / recall / F1 | 0.96 / 0.88 / 0.91 |
| Cross-validation score | 0.974 |

Confusion matrix (rows are actual, columns are predicted):

|  | Predicted retained | Predicted churn |
|---|---|---|
| **Actual retained** | 716 | 5 |
| **Actual churned** | 16 | 113 |

Training and validation accuracy stay close together and level off around 97% over roughly 150 epochs, so the model is not badly overfitting.

## API

Interactive docs are available at `/docs` (Swagger UI) when the server is running.

### `POST /api/getPrediction`: score one customer

Request:

```json
{
  "cid": "customer-123",
  "data": [1, 0, 0.0, 12.4, 3.0, 3.35, 1, 595.9, 315.0, 57.27]
}
```

`data` must contain the 10 features above, in order. Response:

```json
{
  "cid": "customer-123",
  "prediction": 0,
  "probablity": 12.5,
  "data": [1, 0, 0.0, 12.4, 3.0, 3.35, 1, 595.9, 315.0, 57.27]
}
```

`prediction` is 1 for predicted churn and 0 otherwise, and `probablity` is the model output as a percentage. (The key is spelled `probablity` in the current response; clients should use it as written.)

### `POST /api/predictCSV`: score a CSV

Send a CSV file with the 10 feature columns as a `multipart/form-data` upload (field name `file`):

```bash
curl -X POST http://localhost:7860/api/predictCSV \
  -F "file=@dataset/pre_train.csv"
```

The response is a JSON list with one record per row: the original columns plus `pred` (0 or 1) and `prob` (the model output).

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api` | Welcome message, useful as a health check |
| `GET` | `/api/test` | Runs the model over the bundled sample dataset |

CORS is enabled for `http://localhost:3000` so a local web front end can call the API. Add your deployed front-end origin to the `origins` list in `index.py`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn index:app --host 0.0.0.0 --port 7860
```

Then open [http://localhost:7860/docs](http://localhost:7860/docs).

## Run with Docker

```bash
docker build -t lstm-churn .
docker run -p 7860:7860 lstm-churn
```

## Project structure

```
index.py          FastAPI app: preprocessing and prediction endpoints
model/            Trained LSTM model (.h5 and .keras)
dataset/          Sample preprocessed customer data
Dockerfile        Container image (also used by Hugging Face Spaces)
requirements.txt  Pinned dependencies
```

## Known limitations and next steps

- **Feature scaling.** Inputs are currently standardized with a scaler fitted on the data in each request. Fitting and saving the scaler on the training set, then loading it at inference time, would make single-customer predictions consistent with how the model was trained.
- **Input validation.** `data` is not yet checked for exact length, and errors in the CSV endpoint are returned as plain JSON rather than HTTP error codes.
- **Class imbalance.** Only about 15% of the test customers churned, so accuracy alone flatters the model. Watch the churn-class recall (0.88) and consider threshold tuning if missing churners is costly.
- **Ensemble serving.** Only the LSTM is served here. Serving the full voting ensemble, or at least the training code, would make the project reproducible.
