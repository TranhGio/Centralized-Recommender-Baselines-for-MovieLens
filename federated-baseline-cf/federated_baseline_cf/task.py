"""Matrix Factorization training and evaluation for MovieLens 1M."""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from federated_baseline_cf.dataset import load_partition_data
from federated_baseline_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss


# Global cache for dataset metadata
_dataset_cache = {}


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
) -> float:
    """
    Train BasicMF model with MSE loss.

    Args:
        model: BasicMF model instance
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        weight_decay: L2 regularization strength

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

            # Compute loss
            loss = criterion(predictions, ratings)

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
) -> float:
    """
    Train BPRMF model with BPR loss.

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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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
        **kwargs: Additional arguments (weight_decay, num_negatives, etc.)

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


def evaluate_ranking(
    model,
    testloader,
    device: str,
    k_values: list = None,
) -> Dict[str, float]:
    """
    Comprehensive ranking evaluation with multiple metrics.

    Computes for each K in k_values:
        - Hit Rate@K: Fraction of users with at least one hit in top-K
        - Precision@K: Average fraction of relevant items in top-K
        - Recall@K: Average fraction of relevant items retrieved
        - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
        - MRR: Mean Reciprocal Rank (position of first relevant item)
        - Accuracy@K: Same as Hit Rate (binary hit/miss)

    Args:
        model: Model instance (BasicMF or BPRMF)
        testloader: Test data loader
        device: Device ('cuda' or 'cpu')
        k_values: List of K values to evaluate (default: [5, 10, 20])

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

    # Initialize metric accumulators for each K
    metrics_per_k = {k: {
        'hits': 0,
        'precisions': [],
        'recalls': [],
        'ndcgs': [],
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

                # Compute NDCG@K
                ndcg = compute_ndcg(top_k_items, test_items, k)

                metrics_per_k[k]['precisions'].append(precision)
                metrics_per_k[k]['recalls'].append(recall)
                metrics_per_k[k]['ndcgs'].append(ndcg)

            num_users += 1

    # Aggregate metrics
    results = {}

    for k in k_values:
        # Hit Rate@K (also called Accuracy@K in some literature)
        results[f'hit_rate@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0
        results[f'accuracy@{k}'] = results[f'hit_rate@{k}']  # Same metric, different name

        # Precision@K
        results[f'precision@{k}'] = np.mean(metrics_per_k[k]['precisions']) if metrics_per_k[k]['precisions'] else 0.0

        # Recall@K
        results[f'recall@{k}'] = np.mean(metrics_per_k[k]['recalls']) if metrics_per_k[k]['recalls'] else 0.0

        # NDCG@K
        results[f'ndcg@{k}'] = np.mean(metrics_per_k[k]['ndcgs']) if metrics_per_k[k]['ndcgs'] else 0.0

    # MRR (not K-dependent)
    results['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0

    return results
