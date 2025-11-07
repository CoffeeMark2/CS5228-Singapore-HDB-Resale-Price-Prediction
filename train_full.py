import os.path
import time

import numpy as np

from models import *
from validate_kfold import DROP_COLS, VERSION, TARGET, FULL_TRAIN_DATA_PATH

MODEL_PATH = 'models/'


def train_full():
    print("--- Starting Model Training on FULL Data ---")

    CAT_ITERS = 20450
    LGBM_ITERS = 13535
    XGB_ITERS = 20129

    print(f"Using iterations: CAT={CAT_ITERS}, LGBM={LGBM_ITERS}, XGB={XGB_ITERS}")

    try:
        df_full_train = BaseModel.load_data(FULL_TRAIN_DATA_PATH)
    except FileNotFoundError:
        return

    df_full_train[TARGET] = df_full_train['PRICE_PER_SQM'] * df_full_train['FLOOR_AREA_SQM']

    X_train = df_full_train.drop(columns=DROP_COLS, errors='ignore')
    y_train = np.log1p(df_full_train[TARGET])

    print(f"Total features: {X_train.shape[1]} cols.")
    print(f"Total training data size: {len(X_train)} rows.")

    use_gpu = True

    cat_model = CatBoostModel(use_gpu=use_gpu, early_stopping_rounds=None, iterations=CAT_ITERS)
    lgbm_model = LightGBMModel(use_gpu=use_gpu, early_stopping_rounds=None, n_estimators=LGBM_ITERS)
    xgb_model = XGBoostModel(use_gpu=use_gpu, early_stopping_rounds=None, n_estimators=XGB_ITERS)

    start = time.perf_counter()

    cat_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    end = time.perf_counter()
    print(f"Full training completed in {end - start:.2f} seconds.")

    os.makedirs(MODEL_PATH, exist_ok=True)
    cat_model.save_model(os.path.join(MODEL_PATH, f'catboost_model_v{VERSION}.joblib'))
    lgbm_model.save_model(os.path.join(MODEL_PATH, f'lightgbm_model_v{VERSION}.joblib'))
    xgb_model.save_model(os.path.join(MODEL_PATH, f'xgboost_model_v{VERSION}.joblib'))

    print("--- Full Training and saving complete ---")


if __name__ == "__main__":
    train_full()