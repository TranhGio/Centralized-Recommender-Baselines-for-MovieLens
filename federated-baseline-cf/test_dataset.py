"""Test script for MovieLens 1M dataset loading and Dirichlet partitioning."""

from federated_baseline_cf.dataset import (
    download_movielens_1m,
    load_movielens_1m,
    dirichlet_partition_users,
    create_global_mappings,
    load_partition_data,
)


def test_basic_loading():
    """Test basic dataset loading."""
    print("=" * 80)
    print("TEST 1: Basic Dataset Loading")
    print("=" * 80)

    # Download and load
    data_dir = "./data"
    download_movielens_1m(data_dir)
    ratings_df, movies_df, users_df = load_movielens_1m(data_dir)

    print("\nDataset Info:")
    print(f"  Ratings shape: {ratings_df.shape}")
    print(f"  Movies shape: {movies_df.shape}")
    print(f"  Users shape: {users_df.shape}")

    print("\nSample ratings:")
    print(ratings_df.head())

    print("\nSample movies:")
    print(movies_df.head())

    print("\nRating distribution:")
    print(ratings_df["rating"].value_counts().sort_index())


def test_dirichlet_partitioning():
    """Test Dirichlet partitioning."""
    print("\n" + "=" * 80)
    print("TEST 2: Dirichlet Partitioning")
    print("=" * 80)

    # Load data
    ratings_df, movies_df, _ = load_movielens_1m("./data")

    # Test with different alpha values
    for alpha in [0.1, 0.5, 1.0]:
        print(f"\n--- Testing alpha={alpha} (lower = more non-IID) ---")
        partitions = dirichlet_partition_users(
            ratings_df=ratings_df,
            movies_df=movies_df,
            num_clients=10,
            alpha=alpha,
            seed=42,
        )

        # Analyze partitions
        print("\nPartition sizes:")
        for client_id, client_df in partitions.items():
            unique_users = client_df["user_id"].nunique()
            unique_movies = client_df["movie_id"].nunique()
            print(
                f"  Client {client_id}: {len(client_df)} ratings, "
                f"{unique_users} users, {unique_movies} movies"
            )


def test_dataloader():
    """Test DataLoader creation."""
    print("\n" + "=" * 80)
    print("TEST 3: DataLoader Creation")
    print("=" * 80)

    # Load partition data for client 0
    trainloader, testloader, num_users, num_items, user2idx, item2idx = (
        load_partition_data(
            partition_id=0,
            num_partitions=10,
            alpha=0.5,
            test_ratio=0.2,
            batch_size=32,
            data_dir="./data",
        )
    )

    print(f"\nDataset info:")
    print(f"  Total users: {num_users}")
    print(f"  Total items: {num_items}")
    print(f"  Train batches: {len(trainloader)}")
    print(f"  Test batches: {len(testloader)}")
    print(f"  Train samples: {len(trainloader.dataset)}")
    print(f"  Test samples: {len(testloader.dataset)}")

    # Get a sample batch
    print("\nSample batch:")
    for batch in trainloader:
        print(f"  User shape: {batch['user'].shape}")
        print(f"  Item shape: {batch['item'].shape}")
        print(f"  Rating shape: {batch['rating'].shape}")
        print(f"  Sample users: {batch['user'][:5]}")
        print(f"  Sample items: {batch['item'][:5]}")
        print(f"  Sample ratings: {batch['rating'][:5]}")
        break


if __name__ == "__main__":
    # Run all tests
    test_basic_loading()
    test_dirichlet_partitioning()
    test_dataloader()

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
