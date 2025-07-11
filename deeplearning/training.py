from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

sns.set_theme(style="whitegrid")
sns.set_context("poster")
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        patience: int = 5,
        metric_fns: Optional[List[Callable]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = "cpu",
        checkpoint_path: str = "best_model.pt",
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            criterion (Callable): Loss function.
            patience (int): Number of epochs to wait for improvement before early stopping.
            metric_fns (Optional[List[Callable]]): Evaluation metrics.
            scheduler (Optional): Learning rate scheduler.
            device (torch.device): Device to run training on.
            checkpoint_path (str): File path to save the best model.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.scaler = GradScaler() if device.type == "cuda" else None

        try:
            self.criterion = criterion.to(device)
        except Exception:
            print("❗ Non torch criterion.")
            self.criterion = criterion

        if metric_fns is None:
            self.metric_fns: List[Callable] = []
        else:
            self.metric_fns = metric_fns

        self.metric_names = [fn.__name__ for fn in self.metric_fns]

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        for name in self.metric_names:
            self.history[f"train_{name}"] = []
            self.history[f"val_{name}"] = []

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
    ) -> None:
        """
        Trains the model for a given number of epochs.

        Args:
            train_loader (DataLoader): Dataloader for training data.
            val_loader (DataLoader): Dataloader for validation data.
            num_epochs (int): Number of training epochs.
        """
        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(train_loader, epoch, num_epochs)
            val_loss, val_metrics = self.evaluate(val_loader)
            _, train_metrics = self.evaluate(train_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            for name, val_value, train_value in zip(
                self.metric_names,
                val_metrics,
                train_metrics,
            ):
                self.history[f"val_{name}"].append(val_value)
                self.history[f"train_{name}"].append(train_value)

            metrics_str = " | ".join(
                f"Train {name}: {train_value:.4f} | Val {name}: {val_value:.4f}"
                for name, val_value, train_value in zip(
                    self.metric_names,
                    val_metrics,
                    train_metrics,
                )
            )
            print(
                f"📘 Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | {metrics_str}"
            )

            self._check_early_stopping(val_loss)
            if self.early_stop_counter >= self.patience:
                print("⏹️ Early stopping triggered.")
                break

            self._step_scheduler(val_loss)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, List[float]]:
        """
        Evaluates the model on a given dataloader.

        Args:
            dataloader (DataLoader): Dataloader for evaluation.

        Returns:
            Tuple[float, List[float]]: Average loss and list of metric values.
        """
        self.model.eval()
        total_loss = 0.0
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                y_true_all.append(y.cpu())
                y_pred_all.append(outputs.cpu())

        avg_loss = total_loss / len(dataloader.dataset)

        metrics = []
        if self.metric_fns:
            y_true_tensor = torch.cat(y_true_all)
            y_pred_tensor = torch.cat(y_pred_all)
            for fn in self.metric_fns:
                metrics.append(fn(y_true_tensor, y_pred_tensor))

        return avg_loss, metrics

    def plot_epochs(self) -> None:
        """
        Plots training and validation loss and evaluation metrics across epochs.
        """
        if not self.history["train_loss"]:
            print("❗ Training has not been performed yet.")
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Prepare data for seaborn
        data = {
            "Epoch": [],
            "Value": [],
            "Type": [],
            "Metric": [],
        }
        # Loss
        for i, epoch in enumerate(epochs):
            data["Epoch"].append(epoch)
            data["Value"].append(self.history["train_loss"][i])
            data["Type"].append("Train")
            data["Metric"].append("Loss")
            data["Epoch"].append(epoch)
            data["Value"].append(self.history["val_loss"][i])
            data["Type"].append("Val")
            data["Metric"].append("Loss")
        # Metrics
        for name in self.metric_names:
            for i, epoch in enumerate(epochs):
                data["Epoch"].append(epoch)
                data["Value"].append(self.history[f"train_{name}"][i])
                data["Type"].append("Train")
                data["Metric"].append(name)
                data["Epoch"].append(epoch)
                data["Value"].append(self.history[f"val_{name}"][i])
                data["Type"].append("Val")
                data["Metric"].append(name)

        df = pd.DataFrame(data)

        # Plot with seaborn
        g = sns.relplot(
            data=df,
            x="Epoch",
            y="Value",
            hue="Type",
            col="Metric",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=4,
            aspect=1.5,
            palette="tab10",
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Epoch", "Value")
        g.tight_layout()

        for ax in g.axes.flatten():
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

        plt.show()

    def test(self, test_loader: DataLoader) -> None:
        self._load_best_model()
        test_loss, test_metrics = self.evaluate(test_loader)
        metrics_str = " | ".join(
            f"Test {name}: {value:.4f}"
            for name, value in zip(self.metric_names, test_metrics)
        )
        print(f"Test Loss: {test_loss:.4f} | {metrics_str}")

    def _load_best_model(self) -> None:
        print("📦 Loading best model...")
        state_dict = torch.load(self.checkpoint_path, weights_only=True)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval()
        print("✅ Done!")

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """
        Runs one training epoch.

        Args:
            train_loader (DataLoader): Dataloader for training data.
            epoch (int): Current epoch index.
            total_epochs (int): Total number of epochs.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=False,
        )

        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast(device_type=self.device.type):  # TODO: remove?
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            progress_bar.set_postfix(train_loss=loss.item())

        return running_loss / len(train_loader.dataset)

    def _check_early_stopping(self, val_loss: float) -> None:
        """
        Checks and handles early stopping based on validation loss.

        Args:
            val_loss (float): Current validation loss.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print("  🔥 New best model saved!")
        else:
            print("  ❄️ No improvement, model won't be saved!")
            self.early_stop_counter += 1

    def _step_scheduler(self, val_loss: float) -> None:
        """
        Updates the learning rate scheduler if available.

        NOTE: Does not work with ReduceLROnPlateau.
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics=val_loss)
            else:
                self.scheduler.step()
