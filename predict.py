import os

import numpy as np
import pandas as pd

from models import BaseModel
from validate_kfold import DROP_COLS, VERSION

MODEL_PATH = 'models/'
TEST_DATA_PATH = 'data/test_encoded.csv'
OUTPUT_PATH = f'results/predictions_v{VERSION}.csv'


def predict():
    print("--- Starting Predicting ---")

    cat_model = BaseModel.load_model(os.path.join(MODEL_PATH, f'catboost_model_v{VERSION}.joblib'))
    lgbm_model = BaseModel.load_model(os.path.join(MODEL_PATH, f'lightgbm_model_v{VERSION}.joblib'))
    xgb_model = BaseModel.load_model(os.path.join(MODEL_PATH, f'xgboost_model_v{VERSION}.joblib'))

    if not cat_model or not lgbm_model or not xgb_model:
        return

    try:
        df_test = BaseModel.load_data(TEST_DATA_PATH)
    except FileNotFoundError:
        return

    X_test = df_test.drop(columns=DROP_COLS, errors='ignore')

    cat_pred = cat_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    predictions = (cat_pred + lgbm_pred + xgb_pred) / 3
    predictions = np.expm1(predictions)

    output_df = pd.DataFrame({'ID': X_test['ID']-1, 'Predicted': predictions})
    os.makedirs('results/', exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Prediction Complete ---")


if __name__ == "__main__":
    predict()