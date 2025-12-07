# Credit Scoring API Projects

This repository contains a production-ready API for a Credit Scoring model, designed with MLOps best practices.

## Project Structure

```
├── .github/workflows/  # CI/CD Pipeline
├── data/              # Data storage (processed)
├── model/             # Model artifacts
├── notebooks/         # Reference notebooks
├── tests/             # Automated tests
├── src/
│   ├── api/           # FastAPI application
│   ├── dashboard.py   # Streamlit Monitoring Dashboard
│   ├── data_prep.py   # Data processing
│   └── train.py       # Training script
├── Dockerfile         # Container configuration
└── pyproject.toml     # Dependency management
```

## Setup & Installation

The project uses **Poetry** for dependency management.

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Activate Environment**:
    ```bash
    poetry shell
    ```

## Usage

### Running the API
Start the FastAPI server:
```bash
poetry run uvicorn src.api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### API Endpoints
- **GET /health**: Health check. Returns status and model load state.
- **POST /predict**: Predict credit score.
    - **Input**: JSON object with applicant features.
    - **Output**:
        ```json
        {
            "prediction": 0,
            "probability": 0.123
        }
        ```

### Monitoring Dashboard
Run the monitoring dashboard to visualize API usage and data drift:
```bash
poetry run streamlit run src/dashboard.py
```
The dashboard displays:
- **API Performance**: Request counts, distribution of predictions.
- **Data Drift**: Statistical comparison (KS Test) between reference training data and live production data.

## Docker

Build the container:
```bash
docker build -t credit-scoring-api .
```

Run the container:
```bash
docker run -p 8000:8000 credit-scoring-api
```

## Testing & CI/CD

Run automated tests:
```bash
poetry run pytest
```

A **GitHub Actions** workflow (`.github/workflows/ci.yml`) is configured to:
1.  Set up Python 3.12.
2.  Install dependencies via Poetry.
3.  Run the test suite on every push to `main`.
