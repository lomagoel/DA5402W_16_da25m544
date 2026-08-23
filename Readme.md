End-to-End Automated Image Classification PipelineProject Repository: DA5402W_project_16_da25m607 (Replace with your actual repo name)Evaluated Branch: mainTeam 16 MembersPruthwiraj Lenka (Roll No: da25m607) - CI/CD, Docker, FastAPI Deployment, Cloud OpsAmol Goel (Roll No: da25m544) - Data Preprocessing (Beam/Spark), DVC, GitHub ActionsVarun A V (Roll No: da25m633) - Model Training, Ray Integration, Kibana MonitoringNaidu Vamsi Krishna (Roll No: da25m593) - Airflow Orchestration, System Testing, Quality AssuranceProject OverviewThis project implements an automated Machine Learning Operations (MLOps) pipeline designed to continuously combat data drift. Instead of relying on massive static models, this architecture rapidly processes incoming custom data (based on the Caltech-101 dataset), trains lightweight Convolutional Neural Networks (ResNet18, MobileNetV3-Small), evaluates them using F1 Macro scoring to handle class imbalance, and deploys the superior model to a containerized FastAPI endpoint.System Architecturegraph TD
    A[Input Data / Data Changes] -->|Commits| B[GitHub Repository]
    B -->|Triggers CI/CD| C[Apache Airflow Orchestration]
    C -->|Schedules| D[Data Processing Pipeline]
    D -->|Feeds Tensors| E[Model Training & Tuning]
    E -->|Logs Data| F[Metrics & MLflow Evaluation]
    F -->|Threshold Passed| G[FastAPI Inference Serving]
Component BreakdownOrchestration: Apache Airflow manages strict Directed Acyclic Graphs (DAGs) to ensure data preprocessing completes before GPU provisioning.Compute Layer: GCP CPU instances are used for data cleaning/resizing (to prevent GPU idle time), and lightweight 12 GB VRAM GCP GPUs are used for model training.Tracking: MLflow tracks training metrics and model weights.Deployment & CI/CD: GitHub Actions automates container builds and pushes to GCP Artifact Registry. FastAPI serves the final model.Folder Structure and Dependencies.
├── .dvc/                   # Data Version Control configuration
├── .github/workflows/      # GitHub Actions CI/CD pipelines (deploy.yml)
├── preprocessing/          # Data engineering logic
│   ├── Dockerfile          # CPU-optimized container for data prep
│   ├── augment.py
│   └── requirements.txt
├── training/               # Model development and training logic
│   ├── src/
│   │   ├── models/         # ResNet18 and MobileNetV3 architectures
│   │   ├── helpers/        # Ray integration and data loaders
│   │   └── training_config.yaml # Hyperparameters and system config
├── data.dvc                # Dataset tracking pointer
├── docker-compose.yaml     # Local testing environment
└── requirements.txt        # Root dependencies
Setup and Installation InstructionsClone the repository:git clone https://github.com/your-org/DA5402W_project_team16_da25m607.git
cd DA5402W_project_team16_da25m607
Set up the virtual environment:python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Authenticate with Google Cloud:export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/gcp-key.json"
gcloud auth application-default login
Pull the dataset via DVC:dvc pull
Docker Execution Commands1. Build the Preprocessing Container:docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-repo/preprocessor:v1 ./preprocessing
2. Build the Training Container:docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-repo/trainer:v1 -f Dockerfile .
3. Run Locally (Testing):docker run --rm -v $(pwd)/data:/app/data us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-repo/trainer:v1
Steps to Run Pipelines and ServicesThe pipeline is designed to be triggered automatically via GitHub Actions upon a commit to the main branch.To trigger the Airflow pipeline manually via the GCP CLI:gcloud composer environments run YOUR_COMPOSER_ENV_NAME \
    --location us-central1 \
    dags trigger -- caltech101_training_pipeline
To view the MLflow Tracking UI locally:mlflow ui --port 5000
API Usage InstructionsOnce the FastAPI container is successfully deployed and running, you can send inference requests to the /predict endpoint.Request (cURL):curl -X POST "http://[YOUR_SERVER_IP]:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_images/accordion.jpg"
Example JSON Response:{
  "filename": "accordion.jpg",
  "prediction": "accordion",
  "confidence": 0.984,
  "model_version": "v1.2.0"
}
