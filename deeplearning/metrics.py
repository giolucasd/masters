import torch
from sklearn.metrics import cohen_kappa_score
from torch import Tensor

binary_classes = (0, 1)


def accuracy_fn(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the accuracy between predicted and true labels.

    Args:
        y_true (Tensor): Ground-truth labels of shape (N,)
        y_pred (Tensor): Predicted logits or probabilities of shape (N, C)

    Returns:
        float: Accuracy score as a float between 0 and 1
    """
    y_pred_labels = y_pred.argmax(dim=1)
    acc_tensor = (y_pred_labels == y_true).float().mean()
    return acc_tensor.item()


def cohen_kappa_fn(y_true: Tensor, y_pred: Tensor, num_classes: int = 2) -> float:
    """
    Computes the Cohen's Kappa score between predicted and true labels.

    This function safely handles cases where there is only one class present
    in either the ground truth or the predictions.

    Args:
        y_true (Tensor): Ground-truth labels of shape (N,)
        y_pred (Tensor): Predicted logits or probabilities of shape (N, C)

    Returns:
        float: Cohen's Kappa score in the range [-1, 1]
               Returns 0.0 if either y_true or y_pred has only one class.
    """
    y_pred_labels = y_pred.argmax(dim=1)

    unique_true = torch.unique(y_true)
    unique_pred = torch.unique(y_pred_labels)

    if unique_true.numel() < 2:
        print("❗ Only one unique class in y_true. Kappa score set to 0.0.")
        return 0.0
    if unique_pred.numel() < 2:
        print("❗ Only one unique class in y_pred. Kappa score set to 0.0.")
        return 0.0

    kappa = cohen_kappa_score(
        y_true.cpu().numpy(),
        y_pred_labels.cpu().numpy(),
        labels=tuple(range(num_classes)),
    )
    return kappa
