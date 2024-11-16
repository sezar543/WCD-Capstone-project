# update_mlflow_paths.sh


# Replace all instances of the old host path with the new container path

#!/bin/bash
db_path="/app/deploy_mlflow/mlruns/mlflow.db"

# Update the paths in the mlflow.db file (make sure the path is correct)
sqlite3 $db_path <<EOF
UPDATE model_versions 
SET source = REPLACE(source, 'file:///C:/Dell15/p/IMAGE-CLASSIFICATION-DEPLOY/deploy_mlflow/mlruns', 'file:///app/deploy_mlflow/mlruns'),
    storage_location = REPLACE(storage_location, 'file:///C:/Dell15/p/IMAGE-CLASSIFICATION-DEPLOY/deploy_mlflow/mlruns', 'file:///app/deploy_mlflow/mlruns');
EOF