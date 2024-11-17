import sqlite3
import os

# Connect to the SQLite database inside the Docker container
# conn = sqlite3.connect('C:\Dell15\p\IMAGE-CLASSIFICATION-DEPLOY\deploy_mlflow\mlruns\mlflow.db')
## log_file = os.path.join("C:", "Dell15", "p", "IMAGE-CLASSIFICATION-DEPLOY", "deploy_mlflow", "mlruns", "test_sqlite.log")
# log_file = r"C:\Dell15\p\IMAGE-CLASSIFICATION-DEPLOY\deploy_mlflow\mlruns\test_sqlite.log"

conn = sqlite3.connect('/app/deploy_mlflow/mlruns/mlflow.db')
log_file = "/app/deploy_mlflow/test_sqlite.log"

print("****Inside test_sqlite.py****")
print("Log file path:", log_file)  # Print path to verify it's correct


# Ensure the directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "w") as log:
    log.write("****Inside test_sqlite.py****\n")

    cursor = conn.cursor()
    # Query to check registered model versions
    cursor.execute("SELECT * FROM model_versions;")
    model_versions = cursor.fetchall()
    print("Model versions in the SQLite database:")

    log.write("Model versions in the SQLite database:\n")
    for version in model_versions:
        log.write(f"{version}\n")
        print(version)

# Close the connection
conn.close()
