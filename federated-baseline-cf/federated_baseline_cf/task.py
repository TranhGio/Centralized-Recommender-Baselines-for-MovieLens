"""Matrix Factorization training and evaluation for MovieLens 1M."""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Set, Optional
from collections import Counter

from federated_baseline_cf.dataset import load_partition_data
from federated_baseline_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss


# Global cache for dataset metadata
_dataset_cache = {}

# Global cache for item popularity (computed from training data)
_item_popularity_cache = {}


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: str = None,
):
    """
    Load MovieLens 1M data for a specific partition.

    Args:
        partition_id: ID of this client partition
        num_partitions: Total number of client partitions
        alpha: Dirichlet concentration parameter (0.5 recommended)
        test_ratio: Ratio of test data (default: 0.2)
        batch_size: Batch size for DataLoader
        data_dir: Directory for data storage (defaults to project root data/)

    Returns:
        Tuple of (trainloader, testloader, num_users, num_items, user2idx, item2idx)
    """
    trainloader, testloader, num_users, num_items, user2idx, item2idx = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        test_ratio=test_ratio,
        batch_size=batch_size,
        data_dir=data_dir,
    )

    # Cache metadata for model initialization
    _dataset_cache['num_users'] = num_users
    _dataset_cache['num_items'] = num_items
    _dataset_cache['user2idx'] = user2idx
    _dataset_cache['item2idx'] = item2idx

    return trainloader, testloader


def get_model(
    model_type: str = "bpr",
    num_users: int = None,
    num_items: int = None,
    embedding_dim: int = 64,
    dropout: float = 0.1,
):
    """
    Create a Matrix Factorization model.

    Args:
        model_type: "basic" for BasicMF (MSE), "bpr" for BPRMF
        num_users: Number of users (if None, uses cached value)
        num_items: Number of items (if None, uses cached value)
        embedding_dim: Embedding dimensionality (default: 64)
        dropout: Dropout rate (default: 0.1)

    Returns:
        Model instance (BasicMF or BPRMF)
    """
    # Use cached values if not provided
    if num_users is None:
        num_users = _dataset_cache.get('num_users', 6040)
    if num_items is None:
        num_items = _dataset_cache.get('num_items', 3706)

    if model_type.lower() == "basic":
        model = BasicMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
    elif model_type.lower() == "bpr":
        model = BPRMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_bias=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'basic' or 'bpr'.")

    return model


def train_basic_mf(
    model: BasicMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.0,
    global_params: list = None,
) -> float:
    """
    Train BasicMF model with MSE loss.

    When proximal_mu > 0 and global_params is provided, adds FedProx proximal term:
        loss = MSE_loss + (proximal_mu / 2) * ||w - w_global||^2

    Args:
        model: BasicMF model instance
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        weight_decay: L2 regularization strength
        proximal_mu: FedProx proximal term coefficient (0.0 = standard training)
        global_params: List of global model parameters (required if proximal_mu > 0)

    Returns:
        Average training loss
    """
    model.to(device)
    model.train()

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            # Forward pass
            predictions = model(user_ids, item_ids)

            # Compute base loss (MSE)
            loss = criterion(predictions, ratings)

            # FedProx: Add proximal term if enabled
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_w, global_w in zip(model.parameters(), global_params):
                    proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_bpr_mf(
    model: BPRMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    num_negatives: int = 1,
    proximal_mu: float = 0.0,
    global_params: list = None,
) -> float:
    """
    Train BPRMF model with BPR loss.

    When proximal_mu > 0 and global_params is provided, adds FedProx proximal term:
        loss = BPR_loss + (proximal_mu / 2) * ||w - w_global||^2

    Critical for SOTA performance (RecSys 2024):
        - Proper negative sampling
        - Correct loss implementation
        - Appropriate regularization

    Args:
        model: BPRMF model instance
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        weight_decay: L2 regularization strength
        num_negatives: Number of negative samples per positive
        proximal_mu: FedProx proximal term coefficient (0.0 = standard training)
        global_params: List of global model parameters (required if proximal_mu > 0)

    Returns:
        Average training loss
    """
    model.to(device)
    model.train()

    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Build user_rated_items dictionary for negative sampling
    user_rated_items = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            if u not in user_rated_items:
                user_rated_items[u] = set()
            user_rated_items[u].add(i)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            pos_item_ids = batch['item'].to(device)

            # Sample negative items
            neg_item_ids = model.sample_negatives(
                user_ids,
                pos_item_ids,
                num_negatives=num_negatives,
                user_rated_items=user_rated_items,
                sampling_strategy='uniform',
            )

            # Forward pass
            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)

            # Compute BPR loss
            loss = criterion(pos_scores, neg_scores)

            # FedProx: Add proximal term if enabled
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_w, global_w in zip(model.parameters(), global_params):
                    proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# =============================================================================
# SCAFFOLD Training Functions
# =============================================================================

def train_basic_mf_scaffold(
    model: BasicMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    global_c: list = None,
    local_c: list = None,
) -> Tuple[float, list]:
    """
    Train BasicMF model with SCAFFOLD control variate correction.

    SCAFFOLD (Karimireddy et al., ICML 2020) uses control variates to correct
    for client drift in heterogeneous federated settings.

    Gradient correction: g_corrected = g - c_i + c (where c_i is local, c is global)

    Args:
        model: BasicMF model instance
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        weight_decay: L2 regularization strength
        global_c: Global control variate (list of tensors, same shape as model params)
        local_c: Local control variate (list of tensors, same shape as model params)

    Returns:
        Tuple of (average_loss, delta_c) where delta_c is the control variate update
    """
    model.to(device)
    model.train()

    # Save initial parameters for control variate update (Option II)
    initial_params = [p.detach().clone() for p in model.parameters()]

    # Initialize control variates if not provided
    if global_c is None:
        global_c = [torch.zeros_like(p) for p in model.parameters()]
    if local_c is None:
        local_c = [torch.zeros_like(p) for p in model.parameters()]

    criterion = MSELoss()
    # SCAFFOLD typically uses SGD, but Adam can also work
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            # Forward pass
            predictions = model(user_ids, item_ids)

            # Compute loss
            loss = criterion(predictions, ratings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # SCAFFOLD: Apply gradient correction g = g - c_i + c
            for p, gc, lc in zip(model.parameters(), global_c, local_c):
                if p.grad is not None:
                    p.grad.data.add_(gc.to(device) - lc.to(device))

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1

        total_loss += epoch_loss

    # Compute delta_c (Option II from SCAFFOLD paper)
    # c_i_new = c_i - c + (1/(K*lr)) * (x_initial - x_final)
    # delta_c = c_i_new - c_i = -c + (1/(K*lr)) * (x_initial - x_final)
    delta_c = []
    for p_init, p_final, gc in zip(initial_params, model.parameters(), global_c):
        delta = -gc.to(device) + (1.0 / (total_steps * lr)) * (p_init.to(device) - p_final.data)
        delta_c.append(delta.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, delta_c


def train_bpr_mf_scaffold(
    model: BPRMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    num_negatives: int = 1,
    global_c: list = None,
    local_c: list = None,
) -> Tuple[float, list]:
    """
    Train BPRMF model with SCAFFOLD control variate correction.

    SCAFFOLD (Karimireddy et al., ICML 2020) uses control variates to correct
    for client drift in heterogeneous federated settings.

    Gradient correction: g_corrected = g - c_i + c (where c_i is local, c is global)

    Args:
        model: BPRMF model instance
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        weight_decay: L2 regularization strength
        num_negatives: Number of negative samples per positive
        global_c: Global control variate (list of tensors)
        local_c: Local control variate (list of tensors)

    Returns:
        Tuple of (average_loss, delta_c) where delta_c is the control variate update
    """
    model.to(device)
    model.train()

    # Save initial parameters for control variate update (Option II)
    initial_params = [p.detach().clone() for p in model.parameters()]

    # Initialize control variates if not provided
    if global_c is None:
        global_c = [torch.zeros_like(p) for p in model.parameters()]
    if local_c is None:
        local_c = [torch.zeros_like(p) for p in model.parameters()]

    criterion = BPRLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Build user_rated_items dictionary for negative sampling
    user_rated_items = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            if u not in user_rated_items:
                user_rated_items[u] = set()
            user_rated_items[u].add(i)

    total_loss = 0.0
    num_batches = 0
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            pos_item_ids = batch['item'].to(device)

            # Sample negative items
            neg_item_ids = model.sample_negatives(
                user_ids,
                pos_item_ids,
                num_negatives=num_negatives,
                user_rated_items=user_rated_items,
                sampling_strategy='uniform',
            )

            # Forward pass
            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)

            # Compute BPR loss
            loss = criterion(pos_scores, neg_scores)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # SCAFFOLD: Apply gradient correction g = g - c_i + c
            for p, gc, lc in zip(model.parameters(), global_c, local_c):
                if p.grad is not None:
                    p.grad.data.add_(gc.to(device) - lc.to(device))

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1

        total_loss += epoch_loss

    # Compute delta_c (Option II from SCAFFOLD paper)
    delta_c = []
    for p_init, p_final, gc in zip(initial_params, model.parameters(), global_c):
        delta = -gc.to(device) + (1.0 / (total_steps * lr)) * (p_init.to(device) - p_final.data)
        delta_c.append(delta.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, delta_c


def train_scaffold(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    model_type: str = "bpr",
    global_c: list = None,
    local_c: list = None,
    **kwargs
) -> Tuple[float, list]:
    """
    Unified SCAFFOLD training function for both model types.

    Args:
        model: Model instance (BasicMF or BPRMF)
        trainloader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device ('cuda' or 'cpu')
        model_type: "basic" or "bpr"
        global_c: Global control variate
        local_c: Local control variate
        **kwargs: Additional arguments (weight_decay, num_negatives)

    Returns:
        Tuple of (average_loss, delta_c)
    """
    if model_type.lower() == "basic":
        return train_basic_mf_scaffold(
            model,
            trainloader,
            epochs,
            lr,
            device,
            weight_decay=kwargs.get('weight_decay', 1e-5),
            global_c=global_c,
            local_c=local_c,
        )
    elif model_type.lower() == "bpr":
        return train_bpr_mf_scaffold(
            model,
            trainloader,
            epochs,
            lr,
            device,
            weight_decay=kwargs.get('weight_decay', 1e-5),
            num_negatives=kwargs.get('num_negatives', 1),
            global_c=global_c,
            local_c=local_c,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    model_type: str = "bpr",
    **kwargs
) -> float:
    """
    Unified training function for both model types.

    Args:
        model: Model instance (BasicMF or BPRMF)
        trainloader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device ('cuda' or 'cpu')
        model_type: "basic" or "bpr"
        **kwargs: Additional arguments including:
            - weight_decay: L2 regularization strength
            - num_negatives: Number of negative samples (BPR only)
            - proximal_mu: FedProx proximal term coefficient
            - global_params: Global model parameters for FedProx

    Returns:
        Average training loss
    """
    if model_type.lower() == "basic":
        return train_basic_mf(
            model,
            trainloader,
            epochs,
            lr,
            device,
            weight_decay=kwargs.get('weight_decay', 1e-5),
            proximal_mu=kwargs.get('proximal_mu', 0.0),
            global_params=kwargs.get('global_params', None),
        )
    elif model_type.lower() == "bpr":
        return train_bpr_mf(
            model,
            trainloader,
            epochs,
            lr,
            device,
            weight_decay=kwargs.get('weight_decay', 1e-5),
            num_negatives=kwargs.get('num_negatives', 1),
            proximal_mu=kwargs.get('proximal_mu', 0.0),
            global_params=kwargs.get('global_params', None),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def test(
    model,
    testloader,
    device: str,
    model_type: str = "bpr",
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on test set.

    Computes:
        - Loss (MSE or BPR depending on model type)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)

    Args:
        model: Model instance
        testloader: Test data loader
        device: Device
        model_type: "basic" or "bpr"

    Returns:
        Tuple of (loss, metrics_dict)
        where metrics_dict contains {'rmse': float, 'mae': float}
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in testloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            # Get predictions/scores
            if model_type.lower() == "basic":
                predictions = model(user_ids, item_ids)
                # Clamp to valid rating range [1, 5]
                predictions = torch.clamp(predictions, min=1.0, max=5.0)
            elif model_type.lower() == "bpr":
                # For BPR, get scores (not clamped)
                predictions = model(user_ids, item_ids, neg_item_ids=None)
                # For evaluation, can clamp to rating range
                predictions = torch.clamp(predictions, min=1.0, max=5.0)

            # Compute errors
            squared_errors = (predictions - ratings) ** 2
            absolute_errors = torch.abs(predictions - ratings)

            total_squared_error += squared_errors.sum().item()
            total_absolute_error += absolute_errors.sum().item()
            num_samples += len(ratings)

            # Compute loss based on model type
            if model_type.lower() == "basic":
                criterion = MSELoss()
                loss = criterion(predictions, ratings)
            else:
                # For BPR, use MSE for evaluation (common practice)
                mse = squared_errors.mean()
                loss = mse

            total_loss += loss.item() * len(ratings)

    # Compute metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    rmse = np.sqrt(total_squared_error / num_samples) if num_samples > 0 else 0.0
    mae = total_absolute_error / num_samples if num_samples > 0 else 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
    }

    return avg_loss, metrics


def compute_ndcg(ranked_items, relevant_items, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    NDCG measures ranking quality with position discounting.
    Score = DCG / IDCG where:
    - DCG = sum(rel_i / log2(i+1)) for i in top-K
    - IDCG = ideal DCG (perfect ranking)

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs
        k: Cutoff position

    Returns:
        NDCG@K score (0 to 1, higher is better)
    """
    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            # Relevance = 1 (binary relevance)
            # Position discount: 1/log2(rank+1), where rank starts at 1
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # IDCG calculation (ideal ranking - all relevant items first)
    num_relevant = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    # Normalize
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_mrr(ranked_items, relevant_items):
    """
    Compute Mean Reciprocal Rank (MRR) for a single user.

    MRR = 1 / rank_of_first_relevant_item

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs

    Returns:
        Reciprocal rank (1/rank if hit found, 0 otherwise)
    """
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            return 1.0 / (i + 1)  # i+1 because rank starts at 1
    return 0.0


def compute_ap(ranked_items, relevant_items, k: int) -> float:
    """
    Compute Average Precision at K for a single user.

    AP@K = (1/min(K, |relevant|)) * sum(P(i) * rel(i)) for i in 1..K
    where P(i) is precision at position i, and rel(i) is 1 if item at i is relevant.

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs
        k: Cutoff position

    Returns:
        Average Precision at K (0 to 1, higher is better)
    """
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            hits += 1
            # Precision at this position
            precision_sum += hits / (i + 1)

    # Normalize by minimum of K and number of relevant items
    return precision_sum / min(k, len(relevant_items))


def compute_novelty(
    ranked_items,
    item_popularity: Dict[int, float],
    k: int,
) -> float:
    """
    Compute Novelty at K for a single user's recommendations.

    Novelty = average of -log2(popularity) for recommended items.
    Higher novelty means recommending less popular (more surprising) items.

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        item_popularity: Dict mapping item_id -> popularity (0 to 1)
        k: Cutoff position

    Returns:
        Average novelty score (higher = more novel/surprising)
    """
    if len(ranked_items) == 0:
        return 0.0

    novelties = []
    for item in ranked_items[:k]:
        pop = item_popularity.get(item, 1e-10)  # Avoid log(0)
        # Self-information: -log2(p) where p is popularity
        novelties.append(-np.log2(max(pop, 1e-10)))

    return float(np.mean(novelties)) if novelties else 0.0


def evaluate_ranking(
    model,
    testloader,
    device: str,
    k_values: list = None,
    item_popularity: Optional[Dict[int, float]] = None,
    trainloader=None,
) -> Dict[str, float]:
    """
    Comprehensive ranking evaluation with multiple metrics.

    Computes for each K in k_values:
        - Hit Rate@K: Fraction of users with at least one hit in top-K
        - Precision@K: Average fraction of relevant items in top-K
        - Recall@K: Average fraction of relevant items retrieved
        - F1@K: Harmonic mean of Precision and Recall
        - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
        - MAP@K: Mean Average Precision at K
        - Coverage@K: Fraction of catalog items appearing in recommendations
        - Novelty@K: Average inverse popularity of recommended items
        - MRR: Mean Reciprocal Rank (position of first relevant item)
        - Accuracy@K: Same as Hit Rate (binary hit/miss)

    Args:
        model: Model instance (BasicMF or BPRMF)
        testloader: Test data loader
        device: Device ('cuda' or 'cpu')
        k_values: List of K values to evaluate (default: [5, 10, 20])
        item_popularity: Dict mapping item_id -> popularity (0 to 1).
            If None and trainloader provided, computed from trainloader.
        trainloader: Training data loader (used to compute item_popularity if not provided)

    Returns:
        Dictionary of ranking metrics with keys like:
        - 'hit_rate@5', 'precision@10', 'ndcg@20', 'mrr', etc.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    # Collect test interactions per user
    user_test_items = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            if u not in user_test_items:
                user_test_items[u] = set()
            user_test_items[u].add(i)

    # Compute item popularity from training data if not provided
    if item_popularity is None:
        item_popularity = {}
        if trainloader is not None:
            item_counts = Counter()
            total_interactions = 0
            for batch in trainloader:
                items = batch['item'].numpy()
                item_counts.update(items)
                total_interactions += len(items)
            # Normalize to get popularity (fraction of interactions)
            if total_interactions > 0:
                for item_id, count in item_counts.items():
                    item_popularity[item_id] = count / total_interactions
        # Cache for future use
        _item_popularity_cache.update(item_popularity)

    # Get number of items for coverage calculation
    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)

    # Initialize metric accumulators for each K
    metrics_per_k = {k: {
        'hits': 0,
        'precisions': [],
        'recalls': [],
        'f1s': [],
        'ndcgs': [],
        'aps': [],  # Average Precision scores
        'novelties': [],
        'recommended_items': set(),  # For coverage
    } for k in k_values}

    mrr_scores = []
    num_users = 0
    max_k = max(k_values)

    with torch.no_grad():
        for user_id in user_test_items.keys():
            # Get test items for this user
            test_items = user_test_items[user_id]

            # Get top-MAX_K recommendations (we'll slice for different K)
            top_items, _ = model.recommend(user_id, top_k=max_k, exclude_items=None)

            # Compute MRR (only once per user, independent of K)
            mrr = compute_mrr(top_items, test_items)
            mrr_scores.append(mrr)

            # Compute metrics for each K value
            for k in k_values:
                top_k_items = top_items[:k]

                # Compute hits
                hits_for_user = len(set(top_k_items) & test_items)
                if hits_for_user > 0:
                    metrics_per_k[k]['hits'] += 1

                # Compute precision and recall
                precision = hits_for_user / k if k > 0 else 0
                recall = hits_for_user / len(test_items) if len(test_items) > 0 else 0

                # Compute F1@K (harmonic mean of precision and recall)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                # Compute NDCG@K
                ndcg = compute_ndcg(top_k_items, test_items, k)

                # Compute AP@K (Average Precision)
                ap = compute_ap(top_k_items, test_items, k)

                # Compute Novelty@K
                novelty = compute_novelty(top_k_items, item_popularity, k)

                # Track recommended items for coverage
                metrics_per_k[k]['recommended_items'].update(top_k_items)

                metrics_per_k[k]['precisions'].append(precision)
                metrics_per_k[k]['recalls'].append(recall)
                metrics_per_k[k]['f1s'].append(f1)
                metrics_per_k[k]['ndcgs'].append(ndcg)
                metrics_per_k[k]['aps'].append(ap)
                metrics_per_k[k]['novelties'].append(novelty)

            num_users += 1

    # Aggregate metrics
    results = {}

    for k in k_values:
        # Hit Rate@K (also called Accuracy@K in some literature)
        results[f'hit_rate@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0
        results[f'accuracy@{k}'] = results[f'hit_rate@{k}']  # Same metric, different name

        # Precision@K
        results[f'precision@{k}'] = float(np.mean(metrics_per_k[k]['precisions'])) if metrics_per_k[k]['precisions'] else 0.0

        # Recall@K
        results[f'recall@{k}'] = float(np.mean(metrics_per_k[k]['recalls'])) if metrics_per_k[k]['recalls'] else 0.0

        # F1@K
        results[f'f1@{k}'] = float(np.mean(metrics_per_k[k]['f1s'])) if metrics_per_k[k]['f1s'] else 0.0

        # NDCG@K
        results[f'ndcg@{k}'] = float(np.mean(metrics_per_k[k]['ndcgs'])) if metrics_per_k[k]['ndcgs'] else 0.0

        # MAP@K (Mean Average Precision)
        results[f'map@{k}'] = float(np.mean(metrics_per_k[k]['aps'])) if metrics_per_k[k]['aps'] else 0.0

        # Coverage@K (fraction of catalog items recommended)
        results[f'coverage@{k}'] = len(metrics_per_k[k]['recommended_items']) / num_total_items if num_total_items > 0 else 0.0

        # Novelty@K (average inverse popularity)
        results[f'novelty@{k}'] = float(np.mean(metrics_per_k[k]['novelties'])) if metrics_per_k[k]['novelties'] else 0.0

    # MRR (not K-dependent)
    results['mrr'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    return results
