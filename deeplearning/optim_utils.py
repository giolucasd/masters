from typing import Dict, Optional, Tuple, Union

from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


class WarmupReduceLROnPlateau:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_lr: float,
        target_lr: float,
        plateau_scheduler: ReduceLROnPlateau,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.plateau_scheduler = plateau_scheduler
        self.epoch = 0

    def step(self, val_metric: Optional[float] = None) -> None:
        if self.epoch < self.warmup_epochs:
            lr = (
                self.base_lr
                + (self.target_lr - self.base_lr)
                * (self.epoch + 1)
                / self.warmup_epochs
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            if val_metric is not None:
                self.plateau_scheduler.step(val_metric)
        self.epoch += 1

    def state_dict(self) -> Dict:
        return {"epoch": self.epoch, "plateau": self.plateau_scheduler.state_dict()}

    def load_state_dict(self, state: Dict) -> None:
        self.epoch = state["epoch"]
        self.plateau_scheduler.load_state_dict(state["plateau"])


def build_optimizer_and_scheduler(
    model: Module,
    config: Dict[str, Union[float, str, int]],
    train_loader_len: Optional[int] = None,
    total_epochs: int = 20,
) -> Tuple[Optimizer, Union[LRScheduler, WarmupReduceLROnPlateau]]:
    """
    Creates an optimizer and learning rate scheduler based on the given configuration.

    Args:
        model (Module): The PyTorch model to optimize.
        config (Dict[str, Union[float, str, int]]): Configuration dictionary.
        train_loader_len (Optional[int]): Number of training batches (required for OneCycle).
        total_epochs (int): Total number of training epochs.

    Returns:
        Tuple[Optimizer, Union[LRScheduler, WarmupReduceLROnPlateau]]:
            The optimizer and the configured learning rate scheduler.
    """
    lr: float = config.get("lr", 1e-4)
    wd: float = config.get("weight_decay", 1e-4)
    scheduler_type: str = config.get("scheduler", "plateau")
    warmup_epochs: int = config.get("warmup_epochs", 8)
    max_lr: float = config.get("max_lr", 1e-3)
    eta_min: float = config.get("eta_min", 1e-6)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    if scheduler_type == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=lr / max_lr,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        scheduler: LRScheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    elif scheduler_type == "onecycle":
        if train_loader_len is None:
            raise ValueError("train_loader_len is required for OneCycleLR.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=train_loader_len,
            epochs=total_epochs,
            pct_start=0.1,
            div_factor=max_lr / lr,
            final_div_factor=max_lr / eta_min,
            anneal_strategy="cos",
        )

    elif scheduler_type == "step":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=lr / max_lr,
            total_iters=warmup_epochs,
        )
        step_scheduler = StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.5),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_epochs],
        )

    else:  # Default: ReduceLROnPlateau with warm-up
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=eta_min,
        )
        scheduler = WarmupReduceLROnPlateau(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            base_lr=eta_min,
            target_lr=lr,
            plateau_scheduler=plateau_scheduler,
        )

    return optimizer, scheduler
