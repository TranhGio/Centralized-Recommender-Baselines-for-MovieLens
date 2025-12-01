"""Test script for Matrix Factorization models."""

import torch
from federated_baseline_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss


def test_basic_mf():
    """Test Basic Matrix Factorization model."""
    print("=" * 80)
    print("TEST 1: Basic MF Model")
    print("=" * 80)

    # Model parameters
    num_users = 100
    num_items = 50
    embedding_dim = 32
    batch_size = 16

    # Create model
    model = BasicMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        dropout=0.1,
    )

    print(f"\nModel created:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    ratings = torch.rand(batch_size) * 4 + 1  # Ratings between 1-5

    predictions = model(user_ids, item_ids)

    print(f"\nForward pass:")
    print(f"  Input shape: user_ids={user_ids.shape}, item_ids={item_ids.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5].detach().numpy()}")

    # Test loss
    criterion = MSELoss()
    loss = criterion(predictions, ratings)
    print(f"  MSE Loss: {loss.item():.4f}")

    # Test prediction
    pred = model.predict(user_ids[:5], item_ids[:5])
    print(f"\nPrediction (clamped 1-5): {pred.numpy()}")

    # Test recommendation
    top_items, top_scores = model.recommend(user_id=0, top_k=10)
    print(f"\nTop-10 recommendations for user 0:")
    print(f"  Items: {top_items}")
    print(f"  Scores: {top_scores}")

    print("\n✅ BasicMF test passed!")


def test_bpr_mf():
    """Test BPR Matrix Factorization model."""
    print("\n" + "=" * 80)
    print("TEST 2: BPR-MF Model")
    print("=" * 80)

    # Model parameters
    num_users = 100
    num_items = 50
    embedding_dim = 32
    batch_size = 16

    # Create model
    model = BPRMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        dropout=0.1,
        use_bias=True,
    )

    print(f"\nModel created:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with positive and negative samples
    user_ids = torch.randint(0, num_users, (batch_size,))
    pos_item_ids = torch.randint(0, num_items, (batch_size,))
    neg_item_ids = torch.randint(0, num_items, (batch_size,))

    pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)

    print(f"\nForward pass:")
    print(f"  Input shape: {user_ids.shape}")
    print(f"  Positive scores shape: {pos_scores.shape}")
    print(f"  Negative scores shape: {neg_scores.shape}")
    print(f"  Sample pos scores: {pos_scores[:5].detach().numpy()}")
    print(f"  Sample neg scores: {neg_scores[:5].detach().numpy()}")

    # Test BPR loss
    criterion = BPRLoss()
    loss = criterion(pos_scores, neg_scores)
    print(f"  BPR Loss: {loss.item():.4f}")

    # Test negative sampling
    user_rated_items = {0: {1, 5, 10}, 1: {2, 6, 11}}
    neg_samples = model.sample_negatives(
        user_ids[:2],
        pos_item_ids[:2],
        num_negatives=1,
        user_rated_items=user_rated_items,
    )
    print(f"\nNegative sampling:")
    print(f"  User IDs: {user_ids[:2].numpy()}")
    print(f"  Positive items: {pos_item_ids[:2].numpy()}")
    print(f"  Sampled negatives: {neg_samples.numpy()}")

    # Test multiple negatives
    neg_samples_multi = model.sample_negatives(
        user_ids[:2],
        pos_item_ids[:2],
        num_negatives=4,
        user_rated_items=user_rated_items,
    )
    print(f"  Multiple negatives (4): shape={neg_samples_multi.shape}")
    print(f"    {neg_samples_multi.numpy()}")

    # Test recommendation
    top_items, top_scores = model.recommend(user_id=0, top_k=10)
    print(f"\nTop-10 recommendations for user 0:")
    print(f"  Items: {top_items}")
    print(f"  Scores: {top_scores[:5]}...")  # Show first 5 scores

    print("\n✅ BPRMF test passed!")


def test_with_movielens_shape():
    """Test with MovieLens 1M dataset dimensions."""
    print("\n" + "=" * 80)
    print("TEST 3: MovieLens 1M Dimensions")
    print("=" * 80)

    # MovieLens 1M actual dimensions
    num_users = 6040
    num_items = 3706
    embedding_dim = 64

    print(f"\nCreating models with MovieLens 1M dimensions:")
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    print(f"  Embedding dim: {embedding_dim}")

    # Test BasicMF
    basic_mf = BasicMF(num_users, num_items, embedding_dim)
    basic_params = sum(p.numel() for p in basic_mf.parameters())
    print(f"\nBasicMF:")
    print(f"  Total parameters: {basic_params:,}")
    print(f"  Model size: ~{basic_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test BPRMF
    bpr_mf = BPRMF(num_users, num_items, embedding_dim)
    bpr_params = sum(p.numel() for p in bpr_mf.parameters())
    print(f"\nBPRMF:")
    print(f"  Total parameters: {bpr_params:,}")
    print(f"  Model size: ~{bpr_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test forward pass
    batch_size = 256
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))

    # BasicMF
    predictions = basic_mf(user_ids, item_ids)
    print(f"\nBasicMF forward pass:")
    print(f"  Batch size: {batch_size}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:3].detach().numpy()}")

    # BPRMF
    pos_scores = bpr_mf(user_ids, item_ids, neg_item_ids=None)
    print(f"\nBPRMF forward pass:")
    print(f"  Batch size: {batch_size}")
    print(f"  Output shape: {pos_scores.shape}")
    print(f"  Sample scores: {pos_scores[:3].detach().numpy()}")

    print("\n✅ MovieLens dimensions test passed!")


if __name__ == "__main__":
    # Run all tests
    test_basic_mf()
    test_bpr_mf()
    test_with_movielens_shape()

    print("\n" + "=" * 80)
    print("All model tests completed successfully!")
    print("=" * 80)
    print("\nModels are ready for:")
    print("  1. Centralized training (baseline)")
    print("  2. Federated learning integration with Flower")
    print("  3. Comparison: Basic MF (MSE) vs BPR-MF (ranking)")
