import optuna
import os

db_path = "debug_optuna.db"
if os.path.exists(db_path):
    os.remove(db_path)

storage_url = f"sqlite:///{db_path}"
print(f"Creating study with storage: {storage_url}")

try:
    study = optuna.create_study(storage=storage_url, study_name="test_study")
    print("Study created successfully!")
    print(f"Study name: {study.study_name}")
except Exception as e:
    print(f"Error creating study: {e}")
    import traceback

    traceback.print_exc()
finally:
    if os.path.exists(db_path):
        os.remove(db_path)
