#!/usr/bin/env python3
"""Train and evaluate the LSTM deepfake detection model.

Usage:
    python scripts/train.py --data_dir data/processed --output_dir results

Expects preprocessed data files (X.npy, y.npy) from preprocess.py.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm_model import DeepfakeLSTM  # noqa: E402
from src.models.trainer import Trainer  # noqa: E402
from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_file="logs/train.log")


def plot_training_curves(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'.
        output_dir: Directory to save plots.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "training_curves.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved training curves plot")


def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        output_dir: Directory to save plots.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["REAL", "FAKE"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Display values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )

    fig.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    output_dir: Path,
) -> None:
    """Plot ROC curve.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: Area under the ROC curve.
        output_dir: Directory to save plots.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "roc_curve.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve plot")


def main() -> None:
    """Train and evaluate the deepfake detection model."""
    parser = argparse.ArgumentParser(
        description="Train LSTM model for deepfake detection"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed X.npy and y.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform 5-fold cross-validation",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        logger.error(
            "Preprocessed data not found in %s. "
            "Run scripts/preprocess.py first.",
            data_dir,
        )
        sys.exit(1)

    x_data = np.load(str(x_path))
    y_data = np.load(str(y_path))

    logger.info("Loaded data: X=%s, y=%s", x_data.shape, y_data.shape)
    logger.info(
        "Class distribution: REAL=%d, FAKE=%d",
        int(np.sum(y_data == 0)),
        int(np.sum(y_data == 1)),
    )

    input_size = x_data.shape[2]
    evaluator = Evaluator()

    if args.cross_validate:
        # K-fold cross-validation
        logger.info("Running %d-fold cross-validation", args.n_folds)
        cv_results = evaluator.cross_validate(
            x_data,
            y_data,
            n_folds=args.n_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            input_size=input_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=str(output_dir / "cv_checkpoints"),
        )

        # Save CV results
        with open(str(output_dir / "cv_results.json"), "w") as f:
            json.dump(cv_results, f, indent=2)

        logger.info(
            "CV Results: ACC=%.4f±%.4f | AUC=%.4f±%.4f",
            cv_results["mean_accuracy"][0],
            cv_results["std_accuracy"][0],
            cv_results["mean_roc_auc"][0],
            cv_results["std_roc_auc"][0],
        )

    # Single train/val split for final model and plots
    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=args.val_split,
        stratify=y_data,
        random_state=42,
    )

    logger.info(
        "Train: %d samples | Val: %d samples",
        len(x_train),
        len(x_val),
    )

    # Create model
    model = DeepfakeLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
    )

    logger.info("Model architecture:\n%s", model)

    # Train
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    history = trainer.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    # Evaluate on validation set
    y_prob = trainer.predict(x_val)
    metrics = evaluator.evaluate(y_val, y_prob)

    # Generate plots
    plot_training_curves(history, output_dir)

    cm = evaluator.get_confusion_matrix(y_val, y_prob)
    plot_confusion_matrix(cm, output_dir)

    fpr, tpr, _ = evaluator.get_roc_curve(y_val, y_prob)
    plot_roc_curve(fpr, tpr, metrics["roc_auc"], output_dir)

    # Classification report
    report = evaluator.get_classification_report(y_val, y_prob)
    logger.info("Classification Report:\n%s", report)

    # Save results
    results = {
        "metrics": metrics,
        "model_config": {
            "input_size": input_size,
            "hidden_size": args.hidden_size,
            "epochs_trained": len(history["train_loss"]),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "data_info": {
            "total_samples": len(x_data),
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "input_shape": list(x_data.shape),
        },
    }

    with open(str(output_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(str(output_dir / "classification_report.txt"), "w") as f:
        f.write(report)

    logger.info("=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info("Accuracy:  %.4f", metrics["accuracy"])
    logger.info("ROC-AUC:   %.4f", metrics["roc_auc"])
    if "precision" in metrics:
        logger.info("Precision: %.4f", metrics["precision"])
    if "recall" in metrics:
        logger.info("Recall:    %.4f", metrics["recall"])
    logger.info("Results saved to: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
