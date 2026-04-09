"""Evaluation metrics and cross-validation for deepfake detection."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.models.lstm_model import DeepfakeLSTM
from src.models.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluate deepfake detection model performance.

    Supports:
        - Single split evaluation
        - K-fold cross-validation
        - Confusion matrix
        - ROC curve analysis
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the evaluator.

        Args:
            threshold: Classification threshold for probabilities.
        """
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True labels (N,).
            y_prob: Predicted probabilities (N,).

        Returns:
            Dictionary with accuracy, roc_auc, and other metrics.
        """
        y_pred = (y_prob >= self.threshold).astype(int)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = float(tp)
            metrics["true_negatives"] = float(tn)
            metrics["false_positives"] = float(fp)
            metrics["false_negatives"] = float(fn)

            if tp + fp > 0:
                metrics["precision"] = float(tp / (tp + fp))
            if tp + fn > 0:
                metrics["recall"] = float(tp / (tp + fn))

        logger.info(
            "Evaluation: ACC=%.4f | AUC=%.4f",
            metrics["accuracy"],
            metrics["roc_auc"],
        )

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            y_true: True labels (N,).
            y_prob: Predicted probabilities (N,).

        Returns:
            Confusion matrix as 2D numpy array.
        """
        y_pred = (y_prob >= self.threshold).astype(int)
        return confusion_matrix(y_true, y_pred)

    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve points.

        Args:
            y_true: True labels (N,).
            y_prob: Predicted probabilities (N,).

        Returns:
            Tuple of (fpr, tpr, thresholds).
        """
        return roc_curve(y_true, y_prob)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> str:
        """Generate a text classification report.

        Args:
            y_true: True labels (N,).
            y_prob: Predicted probabilities (N,).

        Returns:
            Formatted classification report string.
        """
        y_pred = (y_prob >= self.threshold).astype(int)
        return classification_report(
            y_true,
            y_pred,
            target_names=["REAL", "FAKE"],
        )

    def cross_validate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        input_size: int = 9,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        checkpoint_dir: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """Perform stratified k-fold cross-validation.

        Args:
            x_data: Feature array (N, seq_len, n_features).
            y_data: Label array (N,).
            n_folds: Number of cross-validation folds.
            epochs: Maximum epochs per fold.
            batch_size: Training batch size.
            input_size: Model input feature dimension.
            hidden_size: LSTM hidden size.
            learning_rate: Learning rate.
            checkpoint_dir: Base directory for fold checkpoints.

        Returns:
            Dictionary with per-fold accuracy, roc_auc lists.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_accuracies: list[float] = []
        fold_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(x_data, y_data), 1):
            logger.info("=== Fold %d/%d ===", fold, n_folds)

            x_train = x_data[train_idx]
            y_train = y_data[train_idx]
            x_val = x_data[val_idx]
            y_val = y_data[val_idx]

            # Create a fresh model for each fold
            model = DeepfakeLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
            )

            fold_ckpt = None
            if checkpoint_dir:
                fold_ckpt = str(Path(checkpoint_dir) / f"fold_{fold}")

            trainer = Trainer(
                model=model,
                learning_rate=learning_rate,
            )

            trainer.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                checkpoint_dir=fold_ckpt,
            )

            # Evaluate
            y_prob = trainer.predict(x_val)
            metrics = self.evaluate(y_val, y_prob)

            fold_accuracies.append(metrics["accuracy"])
            fold_aucs.append(metrics["roc_auc"])

            logger.info(
                "Fold %d: ACC=%.4f | AUC=%.4f",
                fold,
                metrics["accuracy"],
                metrics["roc_auc"],
            )

        results = {
            "fold_accuracy": fold_accuracies,
            "fold_roc_auc": fold_aucs,
            "mean_accuracy": [float(np.mean(fold_accuracies))],
            "std_accuracy": [float(np.std(fold_accuracies))],
            "mean_roc_auc": [float(np.mean(fold_aucs))],
            "std_roc_auc": [float(np.std(fold_aucs))],
        }

        logger.info(
            "CV Results: ACC=%.4f±%.4f | AUC=%.4f±%.4f",
            results["mean_accuracy"][0],
            results["std_accuracy"][0],
            results["mean_roc_auc"][0],
            results["std_roc_auc"][0],
        )

        return results
