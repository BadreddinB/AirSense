# AirSense — FastAPI Documentation

## 1. Overview

AirSense exposes a REST API for checking model availability, retrieving model metadata, and predicting pollutant concentrations for the next time step (`t+1`).

The API is implemented with FastAPI and uses Pydantic for request validation, NumPy and pandas to prepare inference input, and joblib to load the trained model.

## 2. Getting Started

Run the following commands from the project root.

### Activate the virtual environment


```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Check the model artifacts

By default, the API expects these files:

- `models/best_model.joblib`
- `models/model_metadata.json`

The metadata file provides the input feature names (`feature_columns`), output target names (`target_columns`), model name, and any stored evaluation information.

### Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`.

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

### Configure artifact paths

The default artifact paths can be overridden with environment variables:

- `MODEL_PATH`: path to the serialized model file.
- `METADATA_PATH`: path to the model metadata JSON file.

For example:

```bash
MODEL_PATH=/path/to/model.joblib \
METADATA_PATH=/path/to/model_metadata.json \
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The model and metadata are loaded when the application starts. If loading either artifact fails, the application records the loading error. Endpoints that require a loaded model return `503 Service Unavailable`.

## 3. API Endpoints

### `GET /health`

Reports whether the model was loaded successfully.

Example request:

```bash
curl -s http://localhost:8000/health
```

Example response when the model is loaded (`200 OK`):

```json
{
  "status": "ok",
  "model_loaded": true,
  "detail": null
}
```

If model loading fails, the endpoint still returns `200 OK`, but the response contains `"status": "error"`, `"model_loaded": false`, and a `"detail"` string describing the loading error.

**Important:** Check `model_loaded` rather than relying on the HTTP status alone to determine whether the model is ready.

### `GET /model-info`

Returns metadata about the loaded model.

Example request:

```bash
curl -s http://localhost:8000/model-info
```

Example response structure (`200 OK`):

```json
{
  "model_name": "example_model",
  "feature_count": 100,
  "target_columns": [
    "target_1",
    "target_2"
  ],
  "evaluation": []
}
```

The values above are placeholders illustrating the response structure. Actual values depend on `models/model_metadata.json` and the loaded model.

Response fields:

- `model_name`: value from the metadata's `model_name`; if absent, the model's Python class name is used.
- `feature_count`: number of expected input features in `feature_columns`.
- `target_columns`: list of prediction target names from the metadata.
- `evaluation`: evaluation information from metadata, or an empty list if that field is absent.

If the model was not loaded, this endpoint returns `503 Service Unavailable`.

### `POST /predict`

Generates predictions for the next time step (`t+1`) from the supplied feature values.

The request body must contain a `features` object mapping every expected feature name to a numeric value. The exact feature names are defined in `feature_columns` in `models/model_metadata.json`.

Example request structure:

```json
{
  "features": {
    "feature_1": 12.5,
    "feature_2": 8.2
  }
}
```

This is only a format example, not a complete valid request. A valid request must include **all expected features** using their exact names. The API metadata may define 100 features; use the actual `feature_columns` list in `models/model_metadata.json` to construct the full payload.

Save a complete payload as `payload.json`, then call the endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @payload.json
```

Example response structure (`200 OK`):

```json
{
  "predictions": {
    "target_1": 10.2,
    "target_2": 5.7
  },
  "horizon": "t+1",
  "model_name": "example_model"
}
```

Target names and numeric values are illustrative. The actual `predictions` object contains the target names defined in `target_columns`, with values returned by the model.

The API orders input values according to `feature_columns` before inference. Therefore, the order of feature keys in the JSON request does not matter.

## 4. Request Validation and Errors

The request model rejects extra fields at the top level. The prediction endpoint also checks the feature names and values.

| HTTP status | Meaning |
|---|---|
| `200 OK` | Request processed. For `/health`, inspect `model_loaded` to confirm model availability. |
| `422 Unprocessable Entity` | Invalid request body, missing or unexpected feature names, or non-finite feature values. |
| `503 Service Unavailable` | Model or metadata could not be loaded for an endpoint that requires the model. |
| `500 Internal Server Error` | The model prediction output has a shape that does not match the expected target columns. |

### Feature validation

`POST /predict` enforces these rules:

- Every expected feature must be present.
- Unexpected feature names are rejected.
- Values must be finite numbers; `NaN`, positive infinity, and negative infinity are rejected.
- Extra fields outside the `features` object are rejected.

If feature names do not match, the response detail includes `missing_features` and `unexpected_features`.

Example validation error:

```json
{
  "detail": {
    "message": "The provided features do not match the model's expected input features.",
    "missing_features": ["missing_feature"],
    "unexpected_features": ["unknown_feature"]
  }
}
```

The feature names above are placeholders. The API returns the actual missing and unexpected names for the submitted request.

## 5. Scope and Limitations

- `POST /predict` predicts from values supplied in the request; it does not collect sensor readings by itself.
- The prediction horizon is `t+1`.
- The exact input features and output targets are defined in `models/model_metadata.json`.
- `GET /health` reports whether model loading succeeded. It does not independently validate data quality or every external dependency.
- The API loads the model and metadata at application startup; changes to those files require restarting the application to be loaded by a new process.
