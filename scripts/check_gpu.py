import catboost as cb
import xgboost as xgb

print(f"XGBoost version: {xgb.__version__}")
try:
    # Try the newer way
    xgb.XGBClassifier(device="cuda").fit([[0]], [0])
    print("XGBoost GPU available (device='cuda')")
except Exception as e1:
    try:
        xgb.XGBClassifier(tree_method="gpu_hist").fit([[0]], [0])
        print("XGBoost GPU available (tree_method='gpu_hist')")
    except Exception as e2:
        print(f"XGBoost GPU NOT available. Error1: {e1}. Error2: {e2}")

print(f"CatBoost version: {cb.__version__}")
try:
    from catboost import utils

    count = utils.get_gpu_device_count()
    print(f"CatBoost GPU devices found: {count}")
except Exception as e:
    print(f"CatBoost GPU check failed: {e}")
