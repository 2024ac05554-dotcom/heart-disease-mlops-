\# Heart Disease Risk Prediction — MLOps Pipeline



\*\*MLOps Assignment 01 (AIMLCZG523)\*\*

\*\*Student Name - Amit Ashok Saggam\*\*

\*\*Student ID - 2024AC05554\*\*

\## Objective



Build a machine learning classifier to predict the risk of heart disease based on

patient health data (UCI Heart Disease — Cleveland dataset), and deploy the solution

as a cloud-ready, monitored API — demonstrating a complete MLOps lifecycle: data

acquisition, EDA, feature engineering, model training with experiment tracking,

CI/CD, containerization, Kubernetes deployment, and monitoring.



\## Project Structure



├── data/

│   ├── get\_data.py          # Downloads the UCI dataset (falls back gracefully if offline)

│   ├── heart.csv             # Raw downloaded dataset

│   └── heart\_clean.csv       # Cleaned dataset (missing values imputed, target binarized)

├── notebooks/

│   └── 01\_eda.ipynb          # EDA: cleaning, visualizations

├── src/

│   ├── preprocess.py         # Preprocessing pipeline (scaling + one-hot encoding)

│   └── train.py              # Model training, tuning, MLflow logging, model packaging

├── app/

│   └── main.py                # FastAPI serving app (/health, /predict, /metrics)

├── tests/

│   ├── test\_preprocess.py    # Unit tests for preprocessing

│   └── test\_model.py          # Unit tests for the saved model

├── k8s/

│   ├── deployment.yaml        # Kubernetes Deployment (2 replicas, health probes)

│   └── service.yaml           # Kubernetes LoadBalancer Service

├── monitoring/

│   ├── prometheus.yml         # Prometheus scrape config

│   └── README.md              # Monitoring stack setup instructions

├── .github/workflows/ci.yml   # CI/CD: lint -> test -> train

├── models/

│   └── model.pkl               # Final packaged model pipeline (preprocessing + classifier)

├── screenshots/                # Evidence screenshots for the report

├── reports/

│   ├── figures/                 # EDA and evaluation plots

│   └── MLOps\_Assignment01\_Report.docx

├── Dockerfile

└── requirements.txt



\## Setup (clean environment)



```powershell

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

python data\\get\_data.py

```



\## Run the pipeline



```powershell

python src\\train.py              # trains, tunes, logs to MLflow, saves models/model.pkl

pytest tests\\ -v                  # run unit tests

mlflow ui                         # view experiment tracking, http://localhost:5000

```



\## Run the API locally



```powershell

uvicorn app.main:app --reload --port 8000

```



\## Run with Docker



```powershell

docker build -t heart-disease-api .

docker run -d -p 8000:8000 --name heart-api heart-disease-api

```



\## Deploy to Kubernetes (Docker Desktop)



```powershell

kubectl apply -f k8s\\deployment.yaml

kubectl apply -f k8s\\service.yaml

kubectl get pods

kubectl get svc

```



\## Monitoring



See `monitoring/README.md` for Prometheus + Grafana setup instructions.



\## Model Summary



Two classifiers (Logistic Regression, Random Forest) were trained and tuned via

GridSearchCV with 5-fold stratified cross-validation. Logistic Regression was

selected as the champion model based on cross-validation ROC-AUC. Full details,

EDA, and evaluation are in the project report.



\## Report



See `MLOps_Assignment01_Report.docx` for the full write-up: setup

instructions, EDA and modeling choices, experiment tracking summary, architecture

diagram, and CI/CD and deployment workflow screenshots.



