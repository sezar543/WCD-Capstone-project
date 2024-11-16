import os

def get_mlflow_db_path():
    # Check if running inside Docker
    if os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read():
        # Running inside Docker
        return '/app/deploy_mlflow/mlruns/mlflow.db'
    else:
        # Running locally
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'IMAGE-CLASSIFICATION-DEPLOY'))
        mlflow_db_path = os.path.join(project_root, 'deploy_mlflow', 'mlruns', 'mlflow.db')
        return os.path.normpath(mlflow_db_path)
    
# def get_mlflow_db_path():
#     # Adjust the number of directory levels if the current assumption isn't correct
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'IMAGE-CLASSIFICATION-DEPLOY'))
#     # Construct the mlflow.db path relative to the correct project root
#     mlflow_db_path = os.path.join(project_root, 'deploy_mlflow', 'mlruns', 'mlflow.db')
#     return os.path.normpath(mlflow_db_path)
#     # return "C:\\Dell15\\p\\deploy_mlflow\\mlruns\\mlflow.db"

# def get_mlflow_db_path():
#     # Move up one directory from deploy_mlflow to get to the project root
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     # Construct the mlflow.db path relative to the project root
#     mlflow_db_path = os.path.join(project_root, 'deploy_mlflow', 'mlruns', 'mlflow.db')
#     return os.path.normpath(mlflow_db_path)