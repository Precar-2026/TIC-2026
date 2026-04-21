"""
Optimized training script for cardiovascular risk prediction.

This version is based on prior training results:
- XGB and LGBM have the best validation AUC/F1.
- RF shows high overfitting and requires stronger regularization.

Main improvements vs train_model.py:
1. Optuna pruning to stop weak trials earlier.
2. Composite objective: validation quality + overfitting penalty.
3. Narrower, model-specific search spaces informed by previous experiments.
4. Optional threshold tuning on validation to improve F1.
5. Reproducible artifact export compatible with existing visualization notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "si", "s"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class ObjectiveWeights:
    roc_auc: float = 0.70
    f1: float = 0.30
    overfit_penalty: float = 0.15


def get_current_experiment_number(models_dir: str) -> int:
    os.makedirs(models_dir, exist_ok=True)
    current_exp_file = os.path.join(models_dir, ".current_experiment.txt")

    if os.path.exists(current_exp_file):
        try:
            with open(current_exp_file, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            pass

    pattern = re.compile(r"^Experimento(\d+)$")
    experiment_numbers = []
    for item in os.listdir(models_dir):
        path = os.path.join(models_dir, item)
        if os.path.isdir(path):
            m = pattern.match(item)
            if m:
                experiment_numbers.append(int(m.group(1)))

    next_number = max(experiment_numbers) + 1 if experiment_numbers else 1
    with open(current_exp_file, "w", encoding="utf-8") as f:
        f.write(str(next_number))
    return next_number


def reset_experiment_counter(models_dir: str) -> None:
    current_exp_file = os.path.join(models_dir, ".current_experiment.txt")
    if os.path.exists(current_exp_file):
        os.remove(current_exp_file)
        logger.info("Experiment counter reset.")


class OptimizedModelTrainer:
    MODEL_NAMES = {
        "LR": "Logistic Regression",
        "RF": "Random Forest",
        "XGB": "XGBoost",
        "LGBM": "LightGBM",
    }

    RANDOM_STATE = 42

    def __init__(
        self,
        input_dir: str,
        model_name: str,
        models_dir: str,
        n_trials: int = 30,
        cv_folds: int = 5,
        tune_threshold: bool = True,
        drop_uncertain_cases: bool = False,
        uncertainty_quantile: float = 0.10,
        use_smote: bool = False,
        smote_sampling_strategy: float = 1.0,
        resampling_method: str = "none",
        undersample_sampling_strategy: float = 1.0,
    ) -> None:
        self.input_dir = input_dir
        self.model_name = model_name.upper()
        self.models_dir = models_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.tune_threshold = tune_threshold
        self.drop_uncertain_cases = drop_uncertain_cases
        self.uncertainty_quantile = uncertainty_quantile
        requested_method = str(resampling_method).strip().lower()
        if requested_method not in {"none", "smote", "undersample"}:
            raise ValueError(
                "resampling_method must be one of: 'none', 'smote', 'undersample'."
            )

        if use_smote and requested_method not in {"none", "smote"}:
            logger.warning(
                "Se recibió --use_smote junto con --resampling_method=%s. "
                "Por compatibilidad se usará SMOTE.",
                requested_method,
            )

        self.resampling_method = "smote" if use_smote else requested_method
        self.use_smote = self.resampling_method == "smote"
        self.smote_sampling_strategy = smote_sampling_strategy
        self.undersample_sampling_strategy = undersample_sampling_strategy

        if self.model_name not in self.MODEL_NAMES:
            raise ValueError(f"Unsupported model '{self.model_name}'.")

        self.model_full_name = self.MODEL_NAMES[self.model_name]
        self.weights = ObjectiveWeights()

        self.X_train: pd.DataFrame | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: pd.DataFrame | None = None
        self.y_val: np.ndarray | None = None
        self.X_test_snapshot: pd.DataFrame | None = None
        self.y_test_snapshot: np.ndarray | None = None

        self.best_params: Dict[str, Any] | None = None
        self.study: optuna.Study | None = None
        self.best_model: Any = None
        self.best_cv_score: float | None = None
        self.best_threshold: float = 0.5
        self.val_metrics: Dict[str, float] = {}
        self.optuna_total_search_seconds: float | None = None
        self.best_trial_number: int | None = None
        self.best_trial_duration_seconds: float | None = None
        self.best_params_fit_seconds: float | None = None
        self.best_params_eval_seconds: float | None = None
        self.best_params_train_eval_total_seconds: float | None = None

        self.y_val_pred: np.ndarray | None = None
        self.y_val_pred_proba: np.ndarray | None = None
        self.y_train_pred: np.ndarray | None = None
        self.y_train_pred_proba: np.ndarray | None = None

        self.experiment_number: int | None = None
        self.experiment_dir: str | None = None
        self.cleaning_info: Dict[str, Any] = {
            "drop_uncertain_cases": self.drop_uncertain_cases,
            "uncertainty_quantile": self.uncertainty_quantile,
            "removed_uncertain_train_samples": 0,
            "uncertainty_threshold_train": None,
            "use_smote": self.use_smote,
            "smote_sampling_strategy": self.smote_sampling_strategy if self.use_smote else None,
            "resampling_method": self.resampling_method,
            "undersample_sampling_strategy": (
                self.undersample_sampling_strategy
                if self.resampling_method == "undersample"
                else None
            ),
        }

    def _build_resampler(self, y: np.ndarray, force_auto_strategy: bool = False) -> Any | None:
        if self.resampling_method == "none":
            return None

        if self.resampling_method == "smote":
            strategy: float | str
            if force_auto_strategy:
                strategy = "auto"
            else:
                strategy = self._resolve_smote_sampling_strategy(y)
            return SMOTE(sampling_strategy=strategy, random_state=self.RANDOM_STATE)

        if self.resampling_method == "undersample":
            strategy = "auto" if force_auto_strategy else self._resolve_undersample_sampling_strategy(y)
            return RandomUnderSampler(sampling_strategy=strategy, random_state=self.RANDOM_STATE)

        raise ValueError(f"Unsupported resampling method: {self.resampling_method}")

    @staticmethod
    def _is_sampling_ratio_error(exc: ValueError) -> bool:
        msg = str(exc).lower()
        return (
            "sampling_strategy" in msg
            or "remove samples from the minority class" in msg
            or "increase the number of samples in the majority class" in msg
            or "required to generate new sample in the majority class" in msg
        )

    def _fit_model(self, model: Any, X: pd.DataFrame, y: np.ndarray) -> Any:
        resampler = self._build_resampler(y)
        if resampler is None:
            model.fit(X, y)
            return model

        try:
            X_res, y_res = resampler.fit_resample(X, y)
        except ValueError as exc:
            if self._is_sampling_ratio_error(exc):
                logger.warning(
                    "%s falló con la estrategia configurada. Reintentando con sampling_strategy='auto'.",
                    self.resampling_method,
                )
                resampler = self._build_resampler(y, force_auto_strategy=True)
                assert resampler is not None
                X_res, y_res = resampler.fit_resample(X, y)
            else:
                raise

        model.fit(X_res, y_res)
        return model

    def _resolve_smote_sampling_strategy(self, y: np.ndarray) -> float | str:
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)

        if classes.size != 2:
            logger.warning(
                "SMOTE fallback a 'auto': la tarea no es binaria o no tiene dos clases en el subset actual."
            )
            return "auto"

        minority = int(counts.min())
        majority = int(counts.max())
        current_ratio = minority / majority if majority > 0 else 0.0
        desired_ratio = float(self.smote_sampling_strategy)

        # SMOTE no puede reducir minoría; si la razón deseada es <= razón actual, usar auto.
        if desired_ratio <= current_ratio + 1e-6:
            logger.warning(
                "SMOTE sampling_strategy=%.3f incompatible con ratio actual=%.3f. "
                "Se usará sampling_strategy='auto'.",
                desired_ratio,
                current_ratio,
            )
            return "auto"

        return desired_ratio

    def _resolve_undersample_sampling_strategy(self, y: np.ndarray) -> float | str:
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)

        if classes.size != 2:
            logger.warning(
                "Undersample fallback a 'auto': la tarea no es binaria o no tiene dos clases en el subset actual."
            )
            return "auto"

        minority = int(counts.min())
        majority = int(counts.max())
        current_ratio = minority / majority if majority > 0 else 0.0
        desired_ratio = float(self.undersample_sampling_strategy)

        # RandomUnderSampler no puede aumentar la clase mayoritaria.
        if desired_ratio < current_ratio - 1e-6:
            logger.warning(
                "Undersample sampling_strategy=%.3f incompatible con ratio actual=%.3f. "
                "Se usará sampling_strategy='auto'.",
                desired_ratio,
                current_ratio,
            )
            return "auto"

        return desired_ratio

    def _drop_uncertain_training_cases(self) -> None:
        assert self.best_model is not None
        assert self.X_train is not None and self.y_train is not None

        if not hasattr(self.best_model, "predict_proba"):
            logger.warning(
                "Model %s has no predict_proba; uncertain-case removal is skipped.",
                self.model_name,
            )
            return

        train_proba = self.best_model.predict_proba(self.X_train)[:, 1]
        train_uncertainty = np.abs(train_proba - 0.5)
        uncertainty_threshold = float(np.quantile(train_uncertainty, self.uncertainty_quantile))
        keep_mask = train_uncertainty > uncertainty_threshold

        removed_count = int((~keep_mask).sum())
        if removed_count == 0:
            logger.info("No uncertain training rows were removed.")
            self.cleaning_info["removed_uncertain_train_samples"] = 0
            self.cleaning_info["uncertainty_threshold_train"] = uncertainty_threshold
            return

        self.X_train = self.X_train.loc[keep_mask].reset_index(drop=True)
        self.y_train = self.y_train[keep_mask]

        self.cleaning_info["removed_uncertain_train_samples"] = removed_count
        self.cleaning_info["uncertainty_threshold_train"] = uncertainty_threshold

        logger.info(
            "Removed %s uncertain training rows (quantile=%.2f, threshold=%.6f).",
            removed_count,
            self.uncertainty_quantile,
            uncertainty_threshold,
        )

    def load_data(self) -> None:
        logger.info("Loading train/validation data from %s", self.input_dir)
        self.X_train = pd.read_csv(os.path.join(self.input_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.input_dir, "y_train.csv")).values.ravel()
        self.X_val = pd.read_csv(os.path.join(self.input_dir, "X_val.csv"))
        self.y_val = pd.read_csv(os.path.join(self.input_dir, "y_val.csv")).values.ravel()

        test_x_path = os.path.join(self.input_dir, "X_test.csv")
        test_y_path = os.path.join(self.input_dir, "y_test.csv")
        if os.path.exists(test_x_path) and os.path.exists(test_y_path):
            self.X_test_snapshot = pd.read_csv(test_x_path)
            self.y_test_snapshot = pd.read_csv(test_y_path).values.ravel()
            logger.info(
                "Test snapshot detected: %s rows, %s features",
                self.X_test_snapshot.shape[0],
                self.X_test_snapshot.shape[1],
            )

        total = self.X_train.shape[0] + self.X_val.shape[0]
        logger.info(
            "Train: %s rows, Val: %s rows, Features: %s",
            self.X_train.shape[0],
            self.X_val.shape[0],
            self.X_train.shape[1],
        )
        logger.info("Train ratio: %.1f%%", (self.X_train.shape[0] / total) * 100.0)

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.model_name == "LR":
            return LogisticRegression(
                **params,
                random_state=self.RANDOM_STATE,
                max_iter=1500,
                n_jobs=-1,
            )
        if self.model_name == "RF":
            return RandomForestClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
            )
        if self.model_name == "XGB":
            return XGBClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
            )
        if self.model_name == "LGBM":
            return LGBMClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        raise ValueError(f"Model '{self.model_name}' is not implemented.")

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.model_name == "LR":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
            return {
                "C": trial.suggest_float("C", 1e-3, 25.0, log=True),
                "penalty": penalty,
                "solver": solver,
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }

        if self.model_name == "RF":
            # Stronger regularization to reduce overfitting observed in previous runs.
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 18),
                "min_samples_split": trial.suggest_int("min_samples_split", 6, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            }

        if self.model_name == "XGB":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 12),
                "gamma": trial.suggest_float("gamma", 0.0, 4.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            }

        if self.model_name == "LGBM":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 70),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            }

        raise ValueError(f"Search space is missing for {self.model_name}.")

    def _cv_objective_score(self, cv_result: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        mean_auc = float(np.mean(cv_result["test_roc_auc"]))
        mean_f1 = float(np.mean(cv_result["test_f1"]))
        train_auc = float(np.mean(cv_result["train_roc_auc"]))

        gap_auc = max(0.0, train_auc - mean_auc)
        score = (
            self.weights.roc_auc * mean_auc
            + self.weights.f1 * mean_f1
            - self.weights.overfit_penalty * gap_auc
        )

        diagnostics = {
            "cv_auc": mean_auc,
            "cv_f1": mean_f1,
            "train_auc": train_auc,
            "gap_auc": gap_auc,
            "composite_score": score,
        }
        return score, diagnostics

    def objective(self, trial: optuna.Trial) -> float:
        assert self.X_train is not None and self.y_train is not None

        params = self.define_search_space(trial)
        model = self.create_model(params)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.RANDOM_STATE)

        estimator: Any = model
        if self.resampling_method != "none":
            resampler = self._build_resampler(self.y_train)
            assert resampler is not None
            estimator = ImbPipeline(
                steps=[
                    ("resampler", resampler),
                    ("model", model),
                ]
            )

        try:
            cv_result = cross_validate(
                estimator,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring={"roc_auc": "roc_auc", "f1": "f1"},
                return_train_score=True,
                n_jobs=-1,
                error_score="raise",
            )
        except ValueError as exc:
            if self.resampling_method != "none" and self._is_sampling_ratio_error(exc):
                logger.warning(
                    "%s falló en algunos folds con la estrategia configurada. "
                    "Reintentando CV con sampling_strategy='auto'.",
                    self.resampling_method,
                )
                auto_resampler = self._build_resampler(self.y_train, force_auto_strategy=True)
                assert auto_resampler is not None
                estimator = ImbPipeline(
                    steps=[
                        ("resampler", auto_resampler),
                        ("model", model),
                    ]
                )
                cv_result = cross_validate(
                    estimator,
                    self.X_train,
                    self.y_train,
                    cv=cv,
                    scoring={"roc_auc": "roc_auc", "f1": "f1"},
                    return_train_score=True,
                    n_jobs=-1,
                    error_score="raise",
                )
            else:
                raise

        score, diagnostics = self._cv_objective_score(cv_result)
        trial.set_user_attr("cv_auc", diagnostics["cv_auc"])
        trial.set_user_attr("cv_f1", diagnostics["cv_f1"])
        trial.set_user_attr("train_auc", diagnostics["train_auc"])
        trial.set_user_attr("gap_auc", diagnostics["gap_auc"])

        # Enable pruning using the composite score as intermediate signal.
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def optimize_hyperparameters(self) -> None:
        logger.info("Starting optimization for %s", self.model_full_name)
        logger.info("Trials: %s | CV folds: %s", self.n_trials, self.cv_folds)
        if self.resampling_method == "smote":
            logger.info("SMOTE habilitado con sampling_strategy=%.3f", self.smote_sampling_strategy)
        elif self.resampling_method == "undersample":
            logger.info(
                "Undersampling habilitado con sampling_strategy=%.3f",
                self.undersample_sampling_strategy,
            )

        sampler = TPESampler(seed=self.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=max(8, self.n_trials // 6), n_warmup_steps=0)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.model_name}_optimized_v2",
        )

        search_start = time.perf_counter()
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=False)
        self.optuna_total_search_seconds = float(time.perf_counter() - search_start)

        self.best_params = self.study.best_params
        self.best_cv_score = float(self.study.best_value)
        best_trial = self.study.best_trial
        self.best_trial_number = int(best_trial.number)
        if best_trial.duration is not None:
            self.best_trial_duration_seconds = float(best_trial.duration.total_seconds())

        logger.info("Optimization finished.")
        logger.info("Best composite CV score: %.6f", self.best_cv_score)
        logger.info("Optimization total time: %.3f seconds", self.optuna_total_search_seconds)
        logger.info("Best trial number: %s", self.best_trial_number)
        if self.best_trial_duration_seconds is not None:
            logger.info("Best trial duration: %.3f seconds", self.best_trial_duration_seconds)
        logger.info("Best params: %s", self.best_params)

    def _find_best_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        thresholds = np.linspace(0.30, 0.70, 81)
        best_thr = 0.5
        best_f1 = -1.0

        for thr in thresholds:
            y_pred_thr = (y_prob >= thr).astype(int)
            score = f1_score(y_true, y_pred_thr, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

        return best_thr

    def train_and_evaluate(self) -> None:
        assert self.X_train is not None and self.y_train is not None
        assert self.X_val is not None and self.y_val is not None
        assert self.best_params is not None

        logger.info("Training final model with best parameters.")
        train_eval_start = time.perf_counter()
        self.best_model = self.create_model(self.best_params)
        fit_start = time.perf_counter()
        self.best_model = self._fit_model(self.best_model, self.X_train, self.y_train)
        self.best_params_fit_seconds = float(time.perf_counter() - fit_start)

        if self.drop_uncertain_cases:
            logger.warning(
                "ADVERTENCIA: drop_uncertain_cases está activado. Se eliminarán ejemplos de "
                "entrenamiento con probabilidad predicha cercana a 0.5. Esta técnica puede "
                "mejorar métricas artificialmente al reducir la ambigüedad del conjunto de "
                "entrenamiento. Para reportar en tesis, justifique estadísticamente esta "
                "decisión o desactívela (--drop_uncertain_cases no incluido = desactivado)."
            )
            logger.info("Uncertain-case removal is enabled for training data.")
            self._drop_uncertain_training_cases()
            self.best_model = self.create_model(self.best_params)
            fit_start = time.perf_counter()
            self.best_model = self._fit_model(self.best_model, self.X_train, self.y_train)
            self.best_params_fit_seconds = float(time.perf_counter() - fit_start)

        eval_start = time.perf_counter()
        if hasattr(self.best_model, "predict_proba"):
            y_train_proba = self.best_model.predict_proba(self.X_train)[:, 1]
            y_val_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        else:
            y_train_proba = self.best_model.predict(self.X_train)
            y_val_proba = self.best_model.predict(self.X_val)

        if self.tune_threshold:
            self.best_threshold = self._find_best_threshold(self.y_val, y_val_proba)
            logger.info("Best validation threshold selected: %.3f", self.best_threshold)
            logger.info(
                "NOTA: Este umbral fue optimizado sobre el conjunto de VALIDACIÓN. "
                "Para la evaluación final (04_evaluacion.ipynb), aplique este mismo umbral "
                "(%.3f) sobre X_test sin re-optimizarlo. Re-optimizar sobre test infla el F1.",
                self.best_threshold,
            )

        y_train_pred = (y_train_proba >= self.best_threshold).astype(int)
        y_val_pred = (y_val_proba >= self.best_threshold).astype(int)

        self.y_train_pred = y_train_pred
        self.y_train_pred_proba = y_train_proba
        self.y_val_pred = y_val_pred
        self.y_val_pred_proba = y_val_proba

        self.val_metrics = {
            "val_accuracy": accuracy_score(self.y_val, y_val_pred),
            "val_precision": precision_score(self.y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(self.y_val, y_val_pred, zero_division=0),
            "val_f1_score": f1_score(self.y_val, y_val_pred, zero_division=0),
            "val_roc_auc": roc_auc_score(self.y_val, y_val_proba),
            "val_mcc": matthews_corrcoef(self.y_val, y_val_pred),
            "best_threshold": self.best_threshold,
        }
        self.best_params_eval_seconds = float(time.perf_counter() - eval_start)
        self.best_params_train_eval_total_seconds = float(time.perf_counter() - train_eval_start)

    def _overfit_diagnostics(self) -> Dict[str, Any]:
        assert self.y_train is not None and self.y_val is not None
        assert self.y_train_pred is not None and self.y_val_pred is not None
        assert self.y_train_pred_proba is not None and self.y_val_pred_proba is not None

        train_metrics = {
            "train_accuracy": accuracy_score(self.y_train, self.y_train_pred),
            "train_precision": precision_score(self.y_train, self.y_train_pred, zero_division=0),
            "train_recall": recall_score(self.y_train, self.y_train_pred, zero_division=0),
            "train_f1_score": f1_score(self.y_train, self.y_train_pred, zero_division=0),
            "train_roc_auc": roc_auc_score(self.y_train, self.y_train_pred_proba),
            "train_mcc": matthews_corrcoef(self.y_train, self.y_train_pred),
        }

        gap_f1 = train_metrics["train_f1_score"] - self.val_metrics["val_f1_score"]
        gap_auc = train_metrics["train_roc_auc"] - self.val_metrics["val_roc_auc"]

        if gap_f1 >= 0.08 or gap_auc >= 0.08:
            risk = "alto"
        elif gap_f1 >= 0.04 or gap_auc >= 0.04:
            risk = "medio"
        else:
            risk = "bajo"

        return {
            **train_metrics,
            **self.val_metrics,
            "gap_f1_score": gap_f1,
            "gap_roc_auc": gap_auc,
            "overfitting_risk": risk,
        }

    def _get_problematic_samples(self) -> pd.DataFrame:
        assert self.X_val is not None and self.y_val is not None
        assert self.y_val_pred is not None and self.y_val_pred_proba is not None

        df = self.X_val.copy().reset_index(drop=True)
        df["row_id_val"] = np.arange(df.shape[0])
        df["y_true"] = self.y_val
        df["y_pred"] = self.y_val_pred
        df["y_pred_proba"] = self.y_val_pred_proba
        df["is_error"] = (df["y_true"] != df["y_pred"]).astype(int)
        df["pred_confidence"] = np.where(df["y_pred"] == 1, df["y_pred_proba"], 1 - df["y_pred_proba"])
        df["uncertainty"] = np.abs(df["y_pred_proba"] - 0.5)

        high_conf_thr = float(df["pred_confidence"].quantile(0.90))
        low_uncertainty_thr = float(df["uncertainty"].quantile(0.10))

        high_conf_errors = df[(df["is_error"] == 1) & (df["pred_confidence"] >= high_conf_thr)].copy()
        high_conf_errors["problem_type"] = "error_alta_confianza"
        high_conf_errors["problem_score"] = high_conf_errors["pred_confidence"]

        uncertain_cases = df[df["uncertainty"] <= low_uncertainty_thr].copy()
        uncertain_cases["problem_type"] = "caso_incierto"
        uncertain_cases["problem_score"] = 1 - (2 * uncertain_cases["uncertainty"])

        out = pd.concat([high_conf_errors, uncertain_cases], ignore_index=True)
        out = out.drop_duplicates(subset=["row_id_val"])
        out = out.sort_values("problem_score", ascending=False)
        return out

    def save_model_and_artifacts(self) -> None:
        assert self.best_model is not None
        assert self.best_params is not None

        os.makedirs(self.models_dir, exist_ok=True)
        self.experiment_number = get_current_experiment_number(self.models_dir)
        self.experiment_dir = os.path.join(self.models_dir, f"Experimento{self.experiment_number}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        exp_model_path = os.path.join(self.experiment_dir, f"best_model_{self.model_name}.joblib")
        std_model_path = os.path.join(self.models_dir, f"best_model_{self.model_name}.joblib")

        joblib.dump(self.best_model, exp_model_path)
        shutil.copy2(exp_model_path, std_model_path)

        logger.info("Model saved at %s", exp_model_path)
        logger.info("Model copied to %s", std_model_path)

        artifacts_dir = os.path.join(self.experiment_dir, "training_artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        assert self.study is not None
        trials_df = self.study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        trials_df.to_csv(os.path.join(artifacts_dir, f"optuna_trials_{self.model_name}.csv"), index=False)

        metrics_payload = {
            "model": self.model_name,
            "model_full_name": self.model_full_name,
            "cv_folds": self.cv_folds,
            "best_cv_composite_score": self.best_cv_score,
            "optuna_total_search_seconds": self.optuna_total_search_seconds,
            "best_trial_number": self.best_trial_number,
            "best_trial_duration_seconds": self.best_trial_duration_seconds,
            "best_params_fit_seconds": self.best_params_fit_seconds,
            "best_params_eval_seconds": self.best_params_eval_seconds,
            "best_params_train_eval_total_seconds": self.best_params_train_eval_total_seconds,
            **self.val_metrics,
            **self.cleaning_info,
        }
        pd.DataFrame([metrics_payload]).to_csv(
            os.path.join(artifacts_dir, f"val_metrics_{self.model_name}.csv"),
            index=False,
        )

        with open(
            os.path.join(artifacts_dir, f"best_params_{self.model_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        # Guardar el umbral de clasificación en archivo independiente.
        threshold_payload = {
            "model": self.model_name,
            "best_threshold": self.best_threshold,
            "tuned_on": "validation_set",
            "warning": (
                "Aplicar este umbral directamente sobre y_test_proba. "
                "No re-optimizar sobre el conjunto de prueba."
            ),
        }
        with open(
            os.path.join(artifacts_dir, f"threshold_{self.model_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(threshold_payload, f, indent=2, ensure_ascii=False)

        assert self.y_val is not None and self.y_val_pred is not None and self.y_val_pred_proba is not None
        pd.DataFrame(
            {
                "y_true": self.y_val,
                "y_pred": self.y_val_pred,
                "y_pred_proba": self.y_val_pred_proba,
            }
        ).to_csv(os.path.join(artifacts_dir, f"val_predictions_{self.model_name}.csv"), index=False)

        overfit = self._overfit_diagnostics()
        with open(
            os.path.join(artifacts_dir, f"overfitting_diagnostics_{self.model_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(overfit, f, indent=2, ensure_ascii=False)

        problematic = self._get_problematic_samples()
        problematic.to_csv(
            os.path.join(artifacts_dir, f"problematic_validation_samples_{self.model_name}.csv"),
            index=False,
        )

        if self.X_test_snapshot is not None and self.y_test_snapshot is not None:
            snapshot_dir = os.path.join(artifacts_dir, "model_inputs_snapshot")
            os.makedirs(snapshot_dir, exist_ok=True)
            self.X_test_snapshot.to_csv(os.path.join(snapshot_dir, "X_test.csv"), index=False)
            pd.DataFrame({"cardio": self.y_test_snapshot}).to_csv(
                os.path.join(snapshot_dir, "y_test.csv"),
                index=False,
            )

        logger.info("Training artifacts saved at %s", artifacts_dir)

    def print_results(self) -> None:
        logger.info("=" * 80)
        logger.info("RESULTS - %s", self.model_full_name)
        logger.info("=" * 80)
        logger.info("Validation accuracy: %.4f", self.val_metrics["val_accuracy"])
        logger.info("Validation precision: %.4f", self.val_metrics["val_precision"])
        logger.info("Validation recall: %.4f", self.val_metrics["val_recall"])
        logger.info("Validation f1_score: %.4f", self.val_metrics["val_f1_score"])
        logger.info("Validation roc_auc: %.4f", self.val_metrics["val_roc_auc"])
        logger.info("Validation mcc: %.4f", self.val_metrics["val_mcc"])
        logger.info("Best threshold: %.3f", self.val_metrics["best_threshold"])
        if self.optuna_total_search_seconds is not None:
            logger.info("Optuna total search time (s): %.3f", self.optuna_total_search_seconds)
        if self.best_trial_number is not None:
            logger.info("Best trial number: %s", self.best_trial_number)
        if self.best_trial_duration_seconds is not None:
            logger.info("Best trial duration (s): %.3f", self.best_trial_duration_seconds)
        if self.best_params_fit_seconds is not None:
            logger.info("Final fit time with best params (s): %.3f", self.best_params_fit_seconds)
        logger.info("Best params: %s", self.best_params)
        logger.info("=" * 80)

    def run_pipeline(self) -> None:
        self.load_data()
        self.optimize_hyperparameters()
        self.train_and_evaluate()
        self.save_model_and_artifacts()
        self.print_results()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized training with Optuna pruning and overfitting-aware objective."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing X_train.csv, y_train.csv, X_val.csv, y_val.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["LR", "RF", "XGB", "LGBM"],
        help="Model key",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Output models folder",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=80,
        help="Optuna number of trials",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Stratified K-Folds for CV",
    )
    parser.add_argument(
        "--no_threshold_tuning",
        action="store_true",
        help="Disable threshold tuning on validation set",
    )
    parser.add_argument(
        "--reset_experiment",
        action="store_true",
        help="Reset experiment counter before saving",
    )
    parser.add_argument(
        "--drop_uncertain_cases",
        action="store_true",
        help="Remove uncertain training rows (proba near 0.5) and retrain",
    )
    parser.add_argument(
        "--uncertainty_quantile",
        type=float,
        default=0.10,
        help="Quantile of most uncertain training rows to remove (default: 0.10)",
    )
    parser.add_argument(
        "--use_smote",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        metavar="BOOL",
        help="Enable SMOTE only on training folds/data. Supports: --use_smote or --use_smote true/false",
    )
    parser.add_argument(
        "--smote_sampling_strategy",
        type=float,
        default=1.0,
        help="SMOTE sampling strategy (0,1]; 1.0 balances minority to majority",
    )
    parser.add_argument(
        "--resampling_method",
        type=str,
        default="none",
        choices=["none", "smote", "undersample"],
        help=(
            "Resampling method for class imbalance: none, smote, undersample. "
            "--use_smote mantiene compatibilidad y tiene prioridad si se activa."
        ),
    )
    parser.add_argument(
        "--undersample_sampling_strategy",
        type=float,
        default=1.0,
        help="Undersampling ratio (0,1]; 1.0 balances minority to majority by cutting majority class",
    )
    return parser.parse_args()


def validate_input_files(input_dir: str) -> None:
    required = ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv"]
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    missing = [name for name in required if not os.path.exists(os.path.join(input_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")


def main() -> None:
    args = parse_args()

    if args.reset_experiment:
        reset_experiment_counter(args.models_dir)

    validate_input_files(args.input_dir)

    if not 0.0 <= args.uncertainty_quantile < 1.0:
        raise ValueError("--uncertainty_quantile must be in [0.0, 1.0).")
    if not 0.0 < args.smote_sampling_strategy <= 1.0:
        raise ValueError("--smote_sampling_strategy must be in (0.0, 1.0].")
    if not 0.0 < args.undersample_sampling_strategy <= 1.0:
        raise ValueError("--undersample_sampling_strategy must be in (0.0, 1.0].")

    trainer = OptimizedModelTrainer(
        input_dir=args.input_dir,
        model_name=args.model,
        models_dir=args.models_dir,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        tune_threshold=not args.no_threshold_tuning,
        drop_uncertain_cases=args.drop_uncertain_cases,
        uncertainty_quantile=args.uncertainty_quantile,
        use_smote=args.use_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
        resampling_method=args.resampling_method,
        undersample_sampling_strategy=args.undersample_sampling_strategy,
    )
    trainer.run_pipeline()


if __name__ == "__main__":
    main()