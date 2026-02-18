import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from scenetokens.utils.constants import SMALL_EPSILON


def compute_jaccard_index(a: set[int], b: set[int]) -> float:
    """Computes the Jaccard Index (Intersection over Union) between two sets.

    Args:
        a (set[int]): set of integer values.
        b (set[int]): set of integer values.

    Returns:
        jaccard_index (float): the intersection over union value [0, 1].
    """
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0


def compute_hamming_distance(a: list[int], b: list[int], *, return_inverse: bool = False) -> float:
    """Computes the Hamming distance between two categorical vectors.  It sums '1' for every position where elements are
    not equal and '0' otherwise.

    Args:
        a (list or tuple): The first categorical vector.
        b (list or tuple): The second categorical vector.
        return_inverse (bool): If 'True', returns the inverse of the Hamming distance.

    Returns:
        int: The number of mismatched positions, or -1 if lengths are unequal.
    """
    if len(a) != len(b):
        error_message = "Error: Vectors must have the same length."
        raise ValueError(error_message)
    hamming_distance = sum(1 for ai, bi in zip(a, b, strict=False) if ai != bi) / len(a)
    if return_inverse:
        return 1.0 - hamming_distance
    return hamming_distance


def compute_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    mask: torch.Tensor,
    valid_idx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Computes the average error between the valid states of two trajectories.

    Nomenclature:
        B: batch size
        M: number of modes
        T: number of timesteps
        D: trajectory dimensions (usually 2 for x and y)

    Args:
        pred_traj (torch.Tensor(B, M, T, D)): predicted trajectory
        gt_traj (torch.Tensor(B, 1, T, D)): ground truth trajectory
        mask (torch.Tensor(B, 1, T)): valid trajectory datapoints
        valid_idx (torch.Tensor(B)): valid_indeces for computing FDE

    Returns:
        ade (torch.Tensor(B, M)) sum of average errors across the trajectory.
        fde (torch.Tensor(B, M)) final error at the endpoint of the trajectory.
    """
    # ade_dist (B, M, T)
    ade_dist = torch.norm(pred_traj - gt_traj, 2, dim=-1)
    # ade_losses (B, M)
    ade = torch.sum(ade_dist * mask, dim=-1) / torch.sum(mask, dim=-1)
    fde = torch.gather(ade_dist, -1, valid_idx).squeeze(-1)
    return ade, fde


def compute_miss_rate(distances: torch.Tensor, miss_threshold: float = 2.0) -> torch.Tensor:
    """Computes the miss rate of the final distances.

    Nomenclature:
        B: batch size
        M: number of modes

    Args:
        distances (torch.Tensor(B, M)): array of distances
        miss_threshold (float): value for determining of a distances is considered a miss

    Return:
        miss_rate (torch.Tensor(B))
    """
    num_modes = distances.shape[1]
    miss_values = distances > miss_threshold
    return miss_values.sum(axis=-1) / num_modes


def compute_perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the perplexity, the uncertainty of the logits with the target.

    Nomenclature:
        B: batch size
        Q: number of queries
        V: vocabulary size

    Args:
        logits (torch.Tensor(B, Q, V)): model output logits
        target (torch.Tensor(B, Q, 1)): target value.

    Returns:
        torch.Tensor(B, 1): perplexity score.
    """
    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the correct target tokens. The gather method will pick the log probabilities of
    # the true target tokens.
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # Calculate the negative log likelihood
    negative_log_likelihood = -target_log_probs

    # Calculate the mean negative log-likelyhood
    mean_ll = negative_log_likelihood.mean()

    # Calculate perplexity as exp(mean negative log likelihood)
    return torch.exp(mean_ll)


def compute_marginal_pdf(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the marginal probability density function (PDF) given two variables.

    Nomenclature:
        B: batch size
        Q: number of queries
        C: number of classes

    Args:
        x (torch.Tensor(B, Q, C)): a tensor representing a variable.
        y (torch.Tensor(B, Q, C)): a tensor representing a variable.
        sigma (float): standrad deviation.

    Returns:
        pdf (torch.Tensor(B, C)): probability density function of x.
        kernel_values: (torch.Tensor(B, C)): kernel density estimation of x.
    """
    # input shapes are maintained through
    residuals = x - y.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + SMALL_EPSILON
    pdf = pdf / normalization
    return pdf, kernel_values


def compute_joint_pdf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the joint probability density function (PDF) between two variables.

    Args:
        x (torch.Tensor(B, C)): a tensor representing a random variable.
        y (torch.Tensor(B, C)): a tensor representing a random variable.

    Returns:
        pdf (torch.Tensor(B, C) joint probability density between the variables.
    """
    # joint kernel shape: (B, C, C)
    joint_kernel_values = torch.matmul(x.transpose(1, 2), y)
    # normalization shape: (B, 1)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + SMALL_EPSILON
    # pdf shape: (B, C, C)
    return joint_kernel_values / normalization


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Computes the mutual information between the X and Y. I(X;Y) = H(X) + H(Y) - H(X;Y), where H(X) and H(Y) are the
    marginal entropies and H(X;Y) is the conditional entropy. Implementation based on:
    https://github.com/connorlee77/pytorch-mutual-information

    Nomenclature:
        B: batch size
        Q: number of queries
        C: number of classes

    Args:
        x (torch.Tensor(B, C)): probability distribution over the classes.
        y (torch.Tensor(B, 1)): target value.
        normalize (bool): if True it will normalize the mutual information value.

    Returns:
        torch.Tensor(B, 1): perplexity score.
    """
    if x.shape != y.shape:
        error_message = f"Shape of x: {x.shape} != shape of y: {y.shape}"
        raise ValueError(error_message)

    num_dims = x.shape[-1]
    bins = nn.Parameter(torch.linspace(0, 1, num_dims).float(), requires_grad=False).to(x.device)

    # Compute the the marginal distribution between the probability distribution x and a uniform distribution
    pdf_x, kernel_values_x = compute_marginal_pdf(x, bins)
    # Compute the the marginal distribution between the target distribution y and a uniform distribution
    pdf_y, kernel_values_y = compute_marginal_pdf(y, bins)
    # The joint distribution between x and y
    pdf_xy = compute_joint_pdf(kernel_values_x, kernel_values_y)

    # Compute the entropies
    H_x = -torch.sum(pdf_x * torch.log2(pdf_x + SMALL_EPSILON), dim=1)  # noqa: N806
    H_y = -torch.sum(pdf_y * torch.log2(pdf_y + SMALL_EPSILON), dim=1)  # noqa: N806
    H_xy = -torch.sum(pdf_xy * torch.log2(pdf_xy + SMALL_EPSILON), dim=(1, 2))  # noqa: N806

    # Compute the mutual information value
    mutual_information = H_x + H_y - H_xy
    if normalize:
        mutual_information = 2 * mutual_information / (H_x + H_y)
    return mutual_information


def compute_binary_confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Calculates the confusion matrix between predictions and labels.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.

    Returns:
        true_positives (torch.Tensor(B)): True postive counts per sample.
        true_negatives (torch.Tensor(B)): True negative counts per sample.
        false_positives (torch.Tensor(B)): False positive counts per sample.
        false_negatives (torch.Tensor(B)): False negative counts per sample.
    """
    assert predictions.shape == labels.shape, "Shapes of predictions and labels must be the same."

    # Calculating True Positives
    true_positives = ((predictions == 1) & (labels == 1)).sum(dim=-1).float()

    # Calculating True Negatives
    true_negatives = ((predictions == 0) & (labels == 0)).sum(dim=-1).float()

    # Calculating False Positives
    false_positives = ((predictions == 1) & (labels == 0)).sum(dim=-1).float()

    # Calculating False Negatives
    false_negatives = ((predictions == 0) & (labels == 1)).sum(dim=-1).float()

    return true_positives, true_negatives, false_positives, false_negatives


def compute_multiclass_accuracy(
    labels: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, ...]:
    """Computes the precision, recall and F1 scores for multiclass classification.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.
        num_classes (int): number of classes.

    Returns:
        precision (torch.Tensor(B)): Accuracy of positive predictions.
        recall (torch.Tensor(B)): Sensitivity of possitive predictions.
        f1_score (torch.Tensor(B)): Balance between precision and recall.
    """
    assert predictions.shape == labels.shape, "Shapes of predictions and labels must be the same."

    batch_size = labels.shape[0]
    confusion_matrix = torch.zeros((batch_size, num_classes, num_classes), dtype=torch.float32, device=labels.device)
    for i in range(batch_size):
        for target, prediction in zip(labels[i].view(-1), predictions[i].view(-1), strict=False):
            confusion_matrix[i, target.long(), prediction.long()] += 1

    true_positives = confusion_matrix.diagonal(dim1=1, dim2=2)
    false_positives = confusion_matrix.sum(dim=1) - true_positives
    false_negatives = confusion_matrix.sum(dim=2) - true_positives

    precision = true_positives / (true_positives + false_positives + SMALL_EPSILON)
    recall = true_positives / (true_positives + false_negatives + SMALL_EPSILON)
    f1_score = 2 * (precision * recall) / (precision + recall + SMALL_EPSILON)

    return precision.mean(dim=1), recall.mean(dim=1), f1_score.mean(dim=1)


def compute_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Computes the precision, recall and F1 scores.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.

    Returns:
        precision (torch.Tensor(B)): Accuracy of positive predictions.
        recall (torch.Tensor(B)): Sensitivity of possitive predictions.
        f1_score (torch.Tensor(B)): Balance between precision and recall.
    """
    true_positives, _, false_positives, false_negatives = compute_binary_confusion_matrix(labels, predictions)

    # Precision
    precision = true_positives / (true_positives + false_positives + SMALL_EPSILON)

    # Recall
    recall = true_positives / (true_positives + false_negatives + SMALL_EPSILON)

    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + SMALL_EPSILON)
    return precision, recall, f1_score


def compute_collision_rate(  # noqa: PLR0913
    ego_pred_traj: torch.Tensor,
    ego_pred_prob: torch.Tensor,
    ego_index: torch.Tensor,
    others_gt_trajs: torch.Tensor,
    others_gt_trajs_mask: torch.Tensor,
    collision_thresholds: list[float] = [0.1, 0.25, 0.5, 1.0],  # noqa: B006
    *,
    only_best_mode: bool = False,
    separate_by_thresholds: bool = False,
) -> dict[str, torch.Tensor]:
    """Computes the collision rate between the predicted trajectory and other agents' trajectories.

    Nomenclature:
        B: batch size
        M: number of predicted modes
        T: number of timesteps
        D: trajectory dimensions (usually 2 for x and y)
        N: number of other agents in the scene

    Args:
        ego_pred_traj (torch.Tensor(B, M, T, D)): predicted trajectory of the ego agent.
        ego_pred_prob (torch.Tensor(B, M)): predicted probability of each mode for the ego agent.
        ego_index (torch.Tensor(B)): index of the ego agent in the others_gt_trajs tensor.
        others_gt_trajs (torch.Tensor(B, N, T, D)): ground truth trajectories of other agents.
        others_gt_trajs_mask (torch.Tensor(B, N, T)): mask for valid trajectory points of other agents.
        collision_thresholds (list[float]): list of distance thresholds to consider a collision.
        only_best_mode (bool): if True, only considers the best mode for collision calculation.
        separate_by_thresholds (bool): if True, returns a dictionary with collision rates for each threshold, otherwise
            returns a dictionary with a single key "collisionRateAllModes".

    Returns:
        collision_rate (dict[str, torch.Tensor]): dictionary containing the collision rate for each threshold.
    """
    batch_size, num_modes, _, _ = ego_pred_traj.shape

    # Zero out the ego agent's trajectory in the others_gt_trajs tensor to avoid self-collision
    batch_indeces = torch.arange(batch_size, device=ego_pred_traj.device)
    other_agents = others_gt_trajs.clone()
    other_agents[batch_indeces, ego_index] = 0.0
    other_agents_masks = others_gt_trajs_mask.clone()
    other_agents_masks[batch_indeces, ego_index] = False

    # Compute pairwise distances between predicted trajectory and other agents' trajectories
    # distances shape: (B, M, N, T)
    if only_best_mode:
        best_mode_indices = torch.argmax(ego_pred_prob, dim=1)  # shape: (B,)
        ego_pred_traj = ego_pred_traj[batch_indeces, best_mode_indices]  # shape: (B, T, D)
        ego_pred_traj = ego_pred_traj.unsqueeze(1)  # shape: (B, 1, T, D)
    ego = ego_pred_traj[:, :, :, :2].unsqueeze(2)  # (B, M, 1, T, D)
    others = other_agents[:, None, :, :, :2]  # (B, 1, N, T, D)
    distances = torch.norm(ego - others, dim=-1)

    # Invalidate masked GT trajectory points
    distances = distances.masked_fill(~other_agents_masks[:, None, :, :], float("inf"))

    # Check for collisions based on distance thresholds
    thresholds = torch.tensor(collision_thresholds, device=ego_pred_traj.device).view(1, 1, 1, 1, -1)
    # Calculate collision counts: shape (B, M, N, T, num_thresholds)
    collision_counts = distances.unsqueeze(-1) < thresholds
    # Collapse agents and time, each pair of predicted trajectory and other agent is considered a collision if any of
    # the timesteps is a collision, we dont want to double count.
    mode_collisions = collision_counts.any(dim=(2, 3))  # shape (B, M, num_thresholds)

    # Return a dictionary with the collision rate for each threshold
    collision_rate = {}
    if separate_by_thresholds:
        for i, threshold in enumerate(collision_thresholds):
            collision_rate[f"{threshold}"] = mode_collisions[:, :, i].float().mean(dim=1)
    else:
        collision_rate = {"all": (mode_collisions.any(dim=-1)).float().mean(dim=1)}
    return collision_rate
