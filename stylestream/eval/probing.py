"""Linear probing for style leakage analysis.

Trains simple linear classifiers on frozen content features to measure
how much style information (speaker identity, accent, emotion) is
retained. Lower classification accuracy means better destylization.

Paper Table 5 targets:
    - HuBERT L18 raw: Speaker ~86%, Accent ~65%, Emotion ~68%
    - Destylizer (offline): Speaker ~3.5%, Accent ~43.5%, Emotion ~47.6%
    - Destylizer (streaming): Speaker ~3.6%, Accent ~43.5%, Emotion ~47.5%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class ProbingResult:
    """Result of a probing experiment."""

    task: str  # "speaker", "accent", "emotion"
    feature_source: str  # "hubert_l18", "destylizer_offline", "destylizer_streaming"
    accuracy: float
    num_classes: int
    num_samples: int
    num_epochs: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "feature_source": self.feature_source,
            "accuracy": round(self.accuracy, 4),
            "num_classes": self.num_classes,
            "num_samples": self.num_samples,
            "num_epochs": self.num_epochs,
            **self.metadata,
        }


class LinearProbe(nn.Module):
    """Simple linear classifier for probing.

    Features are frozen; only this linear layer is trained.

    Parameters
    ----------
    input_dim : int
        Feature dimension (e.g., 768 for Destylizer).
    num_classes : int
        Number of classes to predict.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input features, shape (batch, input_dim).

        Returns
        -------
        Tensor
            Logits, shape (batch, num_classes).
        """
        return self.linear(x)


class StyleProbing:
    """Style leakage analysis via linear probing.

    Extracts features from a Destylizer (or HuBERT), pools them
    to utterance-level vectors, then trains linear classifiers
    to predict speaker, accent, or emotion labels.

    Parameters
    ----------
    feature_dim : int
        Dimension of content features. Default 768.
    device : str
        Device for training. Default "cuda".
    num_epochs : int
        Training epochs for each probe. Default 20.
    lr : float
        Learning rate. Default 1e-3.
    batch_size : int
        Training batch size. Default 64.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        device: str = "cuda",
        num_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        self.feature_dim = feature_dim
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

    def prepare_features(
        self,
        features: list[torch.Tensor],
        labels: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool frame-level features to utterance-level and prepare dataset.

        Parameters
        ----------
        features : list[Tensor]
            List of (T_i, D) feature tensors (variable length).
        labels : list[int]
            Integer class labels, one per utterance.

        Returns
        -------
        tuple[Tensor, Tensor]
            (pooled_features, labels) with shapes (N, D) and (N,).
        """
        pooled = []
        for feat in features:
            # Mean pooling over time dimension
            if feat.dim() == 2:
                pooled.append(feat.mean(dim=0))
            else:
                pooled.append(feat)

        X = torch.stack(pooled)  # (N, D)
        y = torch.tensor(labels, dtype=torch.long)  # (N,)
        return X, y

    def train_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
    ) -> tuple[LinearProbe, float]:
        """Train a linear probe and return accuracy.

        Parameters
        ----------
        train_features : Tensor
            Shape (N_train, D).
        train_labels : Tensor
            Shape (N_train,).
        val_features : Tensor or None
            Shape (N_val, D). If None, evaluates on train set.
        val_labels : Tensor or None
            Shape (N_val,).

        Returns
        -------
        tuple[LinearProbe, float]
            Trained probe and accuracy on val (or train) set.
        """
        num_classes = int(train_labels.max().item()) + 1
        probe = LinearProbe(self.feature_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

        dataset = TensorDataset(
            train_features.to(self.device), train_labels.to(self.device)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        probe.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                logits = probe(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate
        probe.eval()
        if val_features is not None and val_labels is not None:
            eval_X = val_features.to(self.device)
            eval_y = val_labels.to(self.device)
        else:
            eval_X = train_features.to(self.device)
            eval_y = train_labels.to(self.device)

        with torch.no_grad():
            logits = probe(eval_X)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == eval_y).float().mean().item()

        return probe, accuracy

    def run_probing(
        self,
        task: str,
        feature_source: str,
        features: list[torch.Tensor],
        labels: list[int],
        val_features: list[torch.Tensor] | None = None,
        val_labels: list[int] | None = None,
    ) -> ProbingResult:
        """Run a complete probing experiment.

        Parameters
        ----------
        task : str
            Probing task: "speaker", "accent", or "emotion".
        feature_source : str
            Feature source name (e.g., "hubert_l18", "destylizer_offline").
        features : list[Tensor]
            Training features, each (T_i, D).
        labels : list[int]
            Training labels.
        val_features : list[Tensor] or None
            Validation features.
        val_labels : list[int] or None
            Validation labels.

        Returns
        -------
        ProbingResult
            Probing experiment result.
        """
        logger.info("Running %s probing on %s features...", task, feature_source)

        train_X, train_y = self.prepare_features(features, labels)

        val_X, val_y = None, None
        if val_features is not None and val_labels is not None:
            val_X, val_y = self.prepare_features(val_features, val_labels)

        num_classes = int(train_y.max().item()) + 1
        probe, accuracy = self.train_probe(train_X, train_y, val_X, val_y)

        result = ProbingResult(
            task=task,
            feature_source=feature_source,
            accuracy=accuracy,
            num_classes=num_classes,
            num_samples=len(features),
            num_epochs=self.num_epochs,
        )

        logger.info(
            "Probing %s/%s: accuracy=%.2f%% (%d classes, %d samples)",
            task,
            feature_source,
            accuracy * 100,
            num_classes,
            len(features),
        )
        return result
