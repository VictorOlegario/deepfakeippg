"""Training loop with early stopping for the LSTM model."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import DeepfakeLSTM
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train and evaluate the DeepfakeLSTM model.

    Supports:
        - Train/validation split
        - Early stopping
        - Model checkpointing
        - Learning rate scheduling
    """

    def __init__(
        self,
        model: DeepfakeLSTM,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: LSTM model instance.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight.
            patience: Early stopping patience (epochs without improvement).
            device: Device to train on ('cuda', 'cpu', or None for auto).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.patience = patience

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _create_dataloader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays.

        Args:
            x: Feature array (N, seq_len, n_features).
            y: Label array (N,).
            batch_size: Batch size.
            shuffle: Whether to shuffle data.

        Returns:
            DataLoader instance.
        """
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        """Run validation.

        Args:
            dataloader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        checkpoint_dir: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """Train the model with early stopping.

        Args:
            x_train: Training features (N, seq_len, n_features).
            y_train: Training labels (N,).
            x_val: Validation features.
            y_val: Validation labels.
            epochs: Maximum number of epochs.
            batch_size: Training batch size.
            checkpoint_dir: Directory to save model checkpoints.

        Returns:
            Dictionary with 'train_loss' and 'val_loss' histories.
        """
        train_loader = self._create_dataloader(x_train, y_train, batch_size)
        val_loader = self._create_dataloader(x_val, y_val, batch_size, shuffle=False)

        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training: %d train, %d val samples | device=%s",
            len(x_train),
            len(x_val),
            self.device,
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                if checkpoint_dir:
                    torch.save(
                        self.model.state_dict(),
                        str(ckpt_path / "best_model.pt"),
                    )
            else:
                self.epochs_without_improvement += 1

            if epoch % 5 == 0 or epoch == 1:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                    "lr=%.6f | patience=%d/%d",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    current_lr,
                    self.epochs_without_improvement,
                    self.patience,
                )

            if self.epochs_without_improvement >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Load best model if checkpointed
        if checkpoint_dir:
            best_path = ckpt_path / "best_model.pt"
            if best_path.exists():
                self.model.load_state_dict(
                    torch.load(str(best_path), map_location=self.device, weights_only=True)
                )
                logger.info("Loaded best model (val_loss=%.4f)", self.best_val_loss)

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Generate predictions.

        Args:
            x: Input features (N, seq_len, n_features).
            batch_size: Batch size for inference.

        Returns:
            Predicted probabilities (N,).
        """
        self.model.eval()
        loader = self._create_dataloader(
            x, np.zeros(len(x)), batch_size, shuffle=False
        )

        predictions = []
        with torch.no_grad():
            for x_batch, _ in loader:
                probs = self.model.predict_proba(x_batch)
                predictions.append(probs.cpu().numpy())

        return np.concatenate(predictions)
