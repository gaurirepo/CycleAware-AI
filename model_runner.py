#model_runner.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class ModelRunner:

    def __init__(self, preprocess_with, preprocess_without):
        self.preprocess_with = preprocess_with
        self.preprocess_without = preprocess_without

    def build_model(self, model_name, preprocess):

        if model_name == "linear":
            model = LinearRegression()

        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )

        elif model_name == "xgb":
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbosity=0
            )

        elif model_name == "lgbm":
            model = LGBMRegressor(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=64,
                max_depth=-1,
                min_child_samples=10,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=42,
                verbosity=-1
            )

        elif model_name == "cat":
            model = CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return Pipeline([
            ("prep", preprocess),
            ("model", model)
        ])

    def run_single_model(self, model_name,
                         X_train_w, X_test_w,
                         X_train_wo, X_test_wo,
                         y_train, y_test):

        model_with = self.build_model(model_name, self.preprocess_with)
        model_without = self.build_model(model_name, self.preprocess_without)

        model_with.fit(X_train_w, y_train)
        model_without.fit(X_train_wo, y_train)

        pred_with = model_with.predict(X_test_w)
        pred_without = model_without.predict(X_test_wo)

        return {
            "r2_with": r2_score(y_test, pred_with),
            "r2_without": r2_score(y_test, pred_without),
            "mae_with": mean_absolute_error(y_test, pred_with),
            "mae_without": mean_absolute_error(y_test, pred_without),
            "pred_with": pred_with,
            "pred_without": pred_without
        }

    def cross_validate_model(self, model_name,
                             X_w, X_wo,
                             y,
                             groups,
                             n_splits=5):

        gkf = GroupKFold(n_splits=n_splits)

        r2_with_scores = []
        r2_without_scores = []
        mae_with_scores = []
        mae_without_scores = []

        for train_idx, test_idx in gkf.split(X_w, y, groups):

            X_train_w = X_w.iloc[train_idx]
            X_test_w = X_w.iloc[test_idx]

            X_train_wo = X_wo.iloc[train_idx]
            X_test_wo = X_wo.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model_with = self.build_model(model_name, self.preprocess_with)
            model_without = self.build_model(model_name, self.preprocess_without)

            model_with.fit(X_train_w, y_train)
            model_without.fit(X_train_wo, y_train)

            pred_with = model_with.predict(X_test_w)
            pred_without = model_without.predict(X_test_wo)

            r2_with_scores.append(r2_score(y_test, pred_with))
            r2_without_scores.append(r2_score(y_test, pred_without))

            mae_with_scores.append(mean_absolute_error(y_test, pred_with))
            mae_without_scores.append(mean_absolute_error(y_test, pred_without))

        return {
            "r2_with_mean": np.mean(r2_with_scores),
            "r2_without_mean": np.mean(r2_without_scores),
            "mae_with_mean": np.mean(mae_with_scores),
            "mae_without_mean": np.mean(mae_without_scores),
            "delta_mae_mean": np.mean(mae_without_scores) - np.mean(mae_with_scores)
        }

    def run_all_models(self, models,
                       X_train_w, X_test_w,
                       X_train_wo, X_test_wo,
                       y_train_att, y_test_att,
                       y_train_mem, y_test_mem):

        all_results = {"Attention": {}, "Memory": {}}

        for model in models:
            print(f"\nRunning {model.upper()} Model...")

            att_res = self.run_single_model(
                model,
                X_train_w, X_test_w,
                X_train_wo, X_test_wo,
                y_train_att, y_test_att
            )

            mem_res = self.run_single_model(
                model,
                X_train_w, X_test_w,
                X_train_wo, X_test_wo,
                y_train_mem, y_test_mem
            )

            all_results["Attention"][model] = att_res
            all_results["Memory"][model] = mem_res

        return all_results