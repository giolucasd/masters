import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch import Tensor


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


def f1_fn(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the F1 score between predicted and true labels.

    Args:
        y_true (Tensor): Ground-truth labels of shape (N,)
        y_pred (Tensor): Predicted logits or probabilities of shape (N, C)

    Returns:
        float: F1 score between 0 and 1
    """
    y_pred_labels = y_pred.argmax(dim=1)

    # Handle case with only one class present
    unique_true = torch.unique(y_true)
    unique_pred = torch.unique(y_pred_labels)
    if unique_true.numel() < 2 or unique_pred.numel() < 2:
        print("❗ Not enough classes to compute F1 score. Returning 0.0.")
        return 0.0

    f1 = f1_score(
        y_true.cpu().numpy(),
        y_pred_labels.cpu().numpy(),
        average="binary",  # You can change to "macro" if needed
        pos_label=1,
    )
    return f1


def roc_auc_fn(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the ROC AUC score between predicted and true labels.

    Args:
        y_true (Tensor): Ground-truth labels of shape (N,)
        y_pred (Tensor): Predicted logits or probabilities of shape (N, C)

    Returns:
        float: ROC AUC score between 0 and 1
    """
    if y_pred.size(1) != 2:
        raise ValueError(
            "Expected y_pred to have shape (N, 2) for binary classification."
        )

    y_prob = torch.softmax(y_pred, dim=1)[:, 1]  # Get probability for class 1

    unique_true = torch.unique(y_true)
    if unique_true.numel() < 2:
        print(
            "❗ Only one class present in y_true. ROC AUC is undefined. Returning 0.0."
        )
        return 0.0

    auc = roc_auc_score(y_true.cpu().numpy(), y_prob.cpu().numpy())
    return auc
