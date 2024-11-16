#!/bin/sh
/app/deploy_mlflow/update_mlflow_paths.sh
exec mlflow server --backend-store-uri sqlite:////app/deploy_mlflow/mlruns/mlflow.db --default-artifact-root file:///app/deploy_mlflow/mlruns/artifacts --host 0.0.0.0 --port 5000
