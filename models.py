from abc import ABC, abstractmethod

import category_encoders as ce
import joblib
import lightgbm as lgbm
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from scripts.log_config import *

VERSION = 11
logger = get_logger("validate_kfold", VERSION, per_run=True, rotate=False)


class BaseModel(ABC):

    def __init__(self, early_stopping_rounds=100):
        self.model = None
        self.preprocessor = None
        self.early_stopping_rounds = early_stopping_rounds
        self.features = {
            "drop": ["ID"],
            "low_card_cols": ['REGION', 'SPECIAL_MODEL', 'TX_YEAR'],
            "high_card_cols": ['SUBZONE', 'PLANNING_AREA']
        }
        self.categorical_features = self.features["low_card_cols"] + self.features["high_card_cols"]
        self.numerical_features = []
        self.feature_names_out_ = []

    @staticmethod
    def load_data(filepath):
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        return df

    @abstractmethod
    def _create_preprocessor(self):
        pass

    @abstractmethod
    def _preprocess(self, X, y=None, fit_mode=True):
        pass

    @abstractmethod
    def _fit_model(self, X_train, y_train, X_val, y_val):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Fitting {self.model.__class__.__name__}...")
        X_train_processed = self._preprocess(X_train, y_train, fit_mode=True)

        X_val_processed = None
        if X_val is not None and y_val is not None:
            logger.info("Preparing validation set for early stopping...")
            X_val_processed = self._preprocess(X_val, y_val, fit_mode=False)

        self._fit_model(X_train_processed, y_train, X_val_processed, y_val)

        logger.info("Fit complete.")
        if hasattr(self.model, 'best_iteration_'):
            logger.info(f"Best iteration: {self.model.best_iteration_}")
        elif hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        elif hasattr(self.model, 'best_ntree_limit'):
            logger.info(f"Best ntree_limit: {self.model.best_ntree_limit}")

        return self

    def predict(self, X):
        logger.info(f"Predicting with {self.model.__class__.__name__}...")
        X_processed = self._preprocess(X, fit_mode=False)

        if hasattr(self, 'feature_names_out_') and self.feature_names_out_:
            X_processed = X_processed[self.feature_names_out_]

        return self.model.predict(X_processed)

    def save_model(self, filepath):
        logger.info(f"Saving model and preprocessor to {filepath}...")
        joblib.dump(self, filepath)
        logger.info("Model saved.")

    @staticmethod
    def load_model(filepath):
        logger.info(f"Loading model from {filepath}...")
        try:
            model_instance = joblib.load(filepath)
            logger.info(f"Model {model_instance.__class__.__name__} loaded successfully.")
            return model_instance
        except FileNotFoundError:
            logger.error(f"Error: Model file not found at {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


class CatBoostModel(BaseModel):
    def __init__(self, use_gpu=True, iterations=100000, early_stopping_rounds=100, **model_params):
        super().__init__(early_stopping_rounds=early_stopping_rounds)
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            task_type='GPU' if use_gpu else 'CPU',
            **model_params
        )

    def _create_preprocessor(self):
        self.preprocessor = "native"
        pass

    def _preprocess(self, X, y=None, fit_mode=True):
        X_copy = X.drop(columns=self.features["drop"], errors='ignore')

        if fit_mode:
            self.feature_names_out_ = X_copy.columns.tolist()

        return X_copy[self.feature_names_out_]

    def _fit_model(self, X_train, y_train, X_val, y_val):
        eval_params = {}
        if X_val is not None and y_val is not None:
            eval_params = {
                "eval_set": (X_val, y_val),
                "early_stopping_rounds": self.early_stopping_rounds
            }

        self.model.fit(
            X_train, y_train,
            cat_features=self.categorical_features,
            verbose=False,
            **eval_params
        )


class LightGBMModel(BaseModel):
    def __init__(self, use_gpu=True, n_estimators=100000, early_stopping_rounds=100, **model_params):
        super().__init__(early_stopping_rounds=early_stopping_rounds)
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **model_params
        )

    def _create_preprocessor(self):
        self.preprocessor = "native_dtype"
        pass

    def _preprocess(self, X, y=None, fit_mode=True):
        X_copy = X.drop(columns=self.features["drop"], errors='ignore')

        for col in self.categorical_features:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype('category')

        if fit_mode:
            self.feature_names_out_ = X_copy.columns.tolist()

        return X_copy[self.feature_names_out_]

    def _fit_model(self, X_train, y_train, X_val, y_val):
        eval_params = {}
        if X_val is not None and y_val is not None:
            eval_params = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [lgbm.early_stopping(self.early_stopping_rounds, verbose=False)]
            }

        self.model.fit(
            X_train, y_train,
            **eval_params
        )


class XGBoostModel(BaseModel):
    def __init__(self, use_gpu=True, n_estimators=100000, early_stopping_rounds=100, **model_params):
        super().__init__(early_stopping_rounds=early_stopping_rounds)
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            device='cuda' if use_gpu else 'cpu',
            enable_categorical=False,
            early_stopping_rounds=self.early_stopping_rounds,
            **model_params
        )
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self):
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        target_encoder = ce.CatBoostEncoder(cols=self.features["high_card_cols"],
                                            handle_unknown='value',
                                            sigma=None)

        preprocessor = ColumnTransformer(
            transformers=[
                ('ohe', ohe_encoder, self.features["low_card_cols"]),
                ('te', target_encoder, self.features["high_card_cols"])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        return preprocessor

    def _preprocess(self, X, y=None, fit_mode=True):
        X_copy = X.drop(columns=self.features["drop"], errors='ignore')

        if fit_mode:
            logger.info("Fitting XGB preprocessor (OHE + Target Encoding)...")
            X_processed = self.preprocessor.fit_transform(X_copy, y)
            self.feature_names_out_ = self.preprocessor.get_feature_names_out().tolist()
        else:
            X_processed = self.preprocessor.transform(X_copy)

        X_processed_df = pd.DataFrame(
            X_processed,
            columns=self.feature_names_out_,
            index=X_copy.index
        )
        return X_processed_df

    def _fit_model(self, X_train, y_train, X_val, y_val):
        eval_params = {}
        if X_val is not None and y_val is not None:
            eval_params = {
                "eval_set": [(X_val, y_val)],
                "verbose": False
            }

        self.model.fit(
            X_train, y_train,
            **eval_params
        )

