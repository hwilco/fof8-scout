import torch
from fof8_ml.models import CatBoostClassifierWrapper, XGBoostClassifierWrapper


def test_gpu_detection():
    print(f"Torch CUDA available: {torch.cuda.is_available()}")

    params = {"iterations": 10}
    cb_wrapper = CatBoostClassifierWrapper(random_seed=42, use_gpu=True, **params)
    print(f"CatBoost params: {cb_wrapper.params}")

    xgb_params = {"n_estimators": 10}
    xgb_wrapper = XGBoostClassifierWrapper(random_seed=42, use_gpu=True, **xgb_params)
    print(f"XGBoost params: {xgb_wrapper.params}")

    if torch.cuda.is_available():
        assert cb_wrapper.params.get("task_type") == "GPU"
        assert xgb_wrapper.params.get("device") == "cuda"
        print("Success: GPU parameters correctly set!")
    else:
        print("Warning: CUDA not available, skipping parameter assertion.")


if __name__ == "__main__":
    test_gpu_detection()
