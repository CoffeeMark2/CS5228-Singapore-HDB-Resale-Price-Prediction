import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from models import *
from scripts.log_config import get_logger

VERSION = 11
TARGET = 'RESALE_PRICE'
LOG_PATH = 'logs/validation_log.txt'
FULL_TRAIN_DATA_PATH = 'data/train_encoded.csv'
logger = get_logger("validate_kfold", VERSION, per_run=True, rotate=False)
DROP_COLS = [
    TARGET, 'PRICE_PER_SQM',
    'TX_MONTH_SIN', 'TX_MONTH_COS',
    'LEASE_RATIO', 'LEASE_RATIO_LOG', 'LEASE_RATIO_SQRT',
    'GDP_Growth_Rate', 'coe_change_rate',
    'ROLLING_PRICE_GROWTH_3M_TOWN', 'MONTHS_SINCE_LAST_TX_BLOCK'
]


def validate_kfold(n_splits=5):
    logger.info("--- Starting K-Fold Validation ---")

    try:
        df_full_train = BaseModel.load_data(FULL_TRAIN_DATA_PATH)
    except FileNotFoundError:
        return

    df_full_train[TARGET] = df_full_train['PRICE_PER_SQM'] * df_full_train['FLOOR_AREA_SQM']

    X = df_full_train.drop(columns=DROP_COLS, errors='ignore')
    y = np.log1p(df_full_train[TARGET])

    logger.info(f"Total features: {X.shape[1]} cols.")
    logger.info(f"Total training data size: {len(X)} rows.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros(X.shape[0])
    fold_rmses = []

    cat_iters = []
    lgbm_iters = []
    xgb_iters = []

    start_time = time.perf_counter()

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        use_gpu = True
        stopping_rounds = 300

        cat_model_fold = CatBoostModel(use_gpu=use_gpu, early_stopping_rounds=stopping_rounds)
        lgbm_model_fold = LightGBMModel(use_gpu=use_gpu, early_stopping_rounds=stopping_rounds)
        xgb_model_fold = XGBoostModel(use_gpu=use_gpu, early_stopping_rounds=stopping_rounds)

        logger.info("Fitting models for Fold...")
        cat_model_fold.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        lgbm_model_fold.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        xgb_model_fold.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        if fold == 0:
            logger.info("--- Feature Importances (Fold 0) ---")

            try:
                lgbm_features = lgbm_model_fold.feature_names_out_
                lgbm_importances = lgbm_model_fold.model.feature_importances_
                lgbm_fi = pd.DataFrame({'feature': lgbm_features, 'importance': lgbm_importances})
                lgbm_fi = lgbm_fi.sort_values(by='importance', ascending=True)

                logger.info(f"\n--- LightGBM (Bottom 15) ---\n{lgbm_fi.head(15)}")
                zero_imp_lgbm = lgbm_fi[lgbm_fi['importance'] == 0]['feature'].tolist()
                logger.info(f"LGBM Features with ZERO importance: {zero_imp_lgbm}")

            except Exception as e:
                logger.warning(f"Could not get LGBM FI: {e}")

            try:
                cat_features = cat_model_fold.feature_names_out_
                cat_importances = cat_model_fold.model.get_feature_importance()
                cat_fi = pd.DataFrame({'feature': cat_features, 'importance': cat_importances})
                cat_fi = cat_fi.sort_values(by='importance', ascending=True)

                logger.info(f"\n--- CatBoost (Bottom 15) ---\n{cat_fi.head(15)}")
                zero_imp_cat = cat_fi[cat_fi['importance'] == 0]['feature'].tolist()
                logger.info(f"CatBoost Features with ZERO importance: {zero_imp_cat}")

            except Exception as e:
                logger.warning(f"Could not get CatBoost FI: {e}")

            try:
                xgb_features = xgb_model_fold.preprocessor.get_feature_names_out()
                xgb_importances = xgb_model_fold.model.feature_importances_
                xgb_fi = pd.DataFrame({'feature': xgb_features, 'importance': xgb_importances})
                xgb_fi = xgb_fi.sort_values(by='importance', ascending=True)

                logger.info(f"\n--- XGBoost (Bottom 15) ---\n{xgb_fi.head(15)}")
                zero_imp_xgb = xgb_fi[xgb_fi['importance'] == 0]['feature'].tolist()
                logger.info(f"XGBoost Features with ZERO importance: {zero_imp_xgb}")

            except Exception as e:
                logger.warning(f"Could not get XGB FI: {e}")

        cat_iters.append(cat_model_fold.model.best_iteration_)
        lgbm_iters.append(lgbm_model_fold.model.best_iteration_)
        xgb_iters.append(xgb_model_fold.model.best_iteration)

        logger.info("Predicting on validation set...")
        cat_pred = cat_model_fold.predict(X_val)
        lgbm_pred = lgbm_model_fold.predict(X_val)
        xgb_pred = xgb_model_fold.predict(X_val)

        fold_preds = (cat_pred + lgbm_pred + xgb_pred) / 3
        fold_preds = np.expm1(fold_preds)

        oof_predictions[val_index] = fold_preds

        y_val = np.expm1(y_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, fold_preds))
        fold_rmses.append(fold_rmse)
        logger.info(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

    end_time = time.perf_counter()
    logger.info(f"\nK-Fold training completed in {end_time - start_time:.2f} seconds.")

    y = np.expm1(y)
    total_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"{timestamp} - v{VERSION} - RMSE: {total_rmse:.4f}"
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

    logger.info("\n--- K-Fold Validation Complete ---")
    logger.info(f"Average Fold RMSE: {np.mean(fold_rmses):.4f}")
    logger.info(f"**Overall OOF RMSE: {total_rmse:.4f}**")
    logger.info("----------------------------------\n")

    logger.info(f"Best Iterations for CatBoost per fold: {cat_iters}")
    logger.info(f"Best Iterations for LightGBM per fold: {lgbm_iters}")
    logger.info(f"Best Iterations for XGBoost per fold: {xgb_iters}")

    avg_cat_iter = np.mean(cat_iters)
    avg_lgbm_iter = np.mean(lgbm_iters)
    avg_xgb_iter = np.mean(xgb_iters)

    logger.info("\n--- Optimal Iteration Report ---")
    logger.info(f"Avg CatBoost Iterations (from {n_splits} folds): {avg_cat_iter:.0f}")
    logger.info(f"Avg LightGBM Iterations (from {n_splits} folds): {avg_lgbm_iter:.0f}")
    logger.info(f"Avg XGBoost Iterations (from {n_splits} folds): {avg_xgb_iter:.0f}")
    logger.info("----------------------------------")


if __name__ == "__main__":
    validate_kfold()