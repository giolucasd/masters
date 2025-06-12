from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        patience: int = 5,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = "cpu",
        checkpoint_path: str = "best_model.pt",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__ if metric_fn is not None else "val_metric"
        try:
            self.criterion = criterion.to(device)
        except:  # noqa: E722
            self.criterion = criterion

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            self.metric_name: [],
        }
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
    ) -> None:
        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(train_loader, epoch, num_epochs)
            val_loss, val_metric = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history[self.metric_name].append(val_metric)

            print(
                f"📘 Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | {self.metric_name}: {val_metric:.4f}"
            )

            self._check_early_stopping(val_loss)
            if self.early_stop_counter >= self.patience:
                print("⏹️ Early stopping ativado.")
                break

            self._step_scheduler(val_loss)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
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

        # Cálculo da métrica
        if self.metric_fn is not None:
            y_true_tensor = torch.cat(y_true_all)
            y_pred_tensor = torch.cat(y_pred_all)
            val_metric = self.metric_fn(y_true_tensor, y_pred_tensor)
        else:
            val_metric = 0.0

        return avg_loss, val_metric

    def plot_epochs(self) -> None:
        if not self.history["train_loss"]:
            print("❗ Treinamento ainda não foi executado.")
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        # Plot Métrica
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs,
            self.history[self.metric_name],
            label=self.metric_name,
            color="green",
        )
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_name)
        plt.title(f"{self.metric_name} over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> float:
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
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * x.size(0)
            progress_bar.set_postfix(train_loss=loss.item())

        return running_loss / len(train_loader.dataset)

    def _check_early_stopping(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print("  🔥 Novo melhor modelo salvo!")
        else:
            self.early_stop_counter += 1

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is not None:
            try:
                self.scheduler.step(val_loss)  # ReduceLROnPlateau
            except TypeError:
                self.scheduler.step()  # other schedulers
