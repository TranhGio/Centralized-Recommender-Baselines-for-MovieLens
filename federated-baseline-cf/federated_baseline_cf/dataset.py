"""MovieLens 1M Dataset Loading and Dirichlet Partitioning."""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens ratings."""

    def __init__(self, ratings_df: pd.DataFrame, user2idx: Dict, item2idx: Dict):
        """
        Initialize MovieLens Dataset.

        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
            user2idx: Mapping from user_id to index
            item2idx: Mapping from movie_id to index
        """
        self.ratings = ratings_df
        self.user2idx = user2idx
        self.item2idx = item2idx

        # Convert to indexed format
        self.users = torch.LongTensor(
            [user2idx[uid] for uid in ratings_df["user_id"].values]
        )
        self.items = torch.LongTensor(
            [item2idx[mid] for mid in ratings_df["movie_id"].values]
        )
        self.ratings_tensor = torch.FloatTensor(ratings_df["rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings_tensor[idx],
        }


def download_movielens_1m(data_dir: str = "./data") -> str:
    """
    Download MovieLens 1M dataset.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to the extracted dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ml_dir = data_path / "ml-1m"
    if ml_dir.exists():
        print(f"MovieLens 1M already exists at {ml_dir}")
        return str(ml_dir)

    # Download dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = data_path / "ml-1m.zip"

    print(f"Downloading MovieLens 1M from {url}...")
    urlretrieve(url, zip_path)

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Clean up zip file
    zip_path.unlink()
    print(f"MovieLens 1M downloaded and extracted to {ml_dir}")

    return str(ml_dir)


def load_movielens_1m(data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 1M dataset.

    Args:
        data_dir: Directory containing the ml-1m folder

    Returns:
        Tuple of (ratings_df, movies_df, users_df)
    """
    ml_dir = Path(data_dir) / "ml-1m"

    # Load ratings
    ratings = pd.read_csv(
        ml_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Load movies
    movies = pd.read_csv(
        ml_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    # Load users
    users = pd.read_csv(
        ml_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    print(f"Loaded {len(ratings)} ratings from {len(users)} users on {len(movies)} movies")
    return ratings, movies, users


def compute_user_genre_distribution(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute user's genre preference distribution.

    Args:
        ratings_df: DataFrame with ratings
        movies_df: DataFrame with movie genres

    Returns:
        DataFrame with user_id and genre preference proportions
    """
    # Merge ratings with movies to get genres
    merged = ratings_df.merge(movies_df[["movie_id", "genres"]], on="movie_id")

    # Split genres (format: "Action|Adventure|Sci-Fi")
    all_genres = set()
    for genres_str in movies_df["genres"].unique():
        all_genres.update(genres_str.split("|"))
    all_genres = sorted(all_genres)

    # Compute genre counts per user
    user_genre_counts = {user_id: {genre: 0 for genre in all_genres}
                         for user_id in ratings_df["user_id"].unique()}

    for _, row in merged.iterrows():
        user_id = row["user_id"]
        genres = row["genres"].split("|")
        for genre in genres:
            user_genre_counts[user_id][genre] += 1

    # Convert to DataFrame
    user_genre_df = pd.DataFrame(user_genre_counts).T
    user_genre_df.index.name = "user_id"

    # Normalize to get proportions
    user_genre_proportions = user_genre_df.div(user_genre_df.sum(axis=1), axis=0)
    user_genre_proportions = user_genre_proportions.fillna(0)

    return user_genre_proportions


def dirichlet_partition_users(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    min_ratings_per_client: int = 100,
    seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """
    Partition users across clients using Dirichlet distribution over genre preferences.

    This creates non-IID data distribution where clients have users with similar
    genre preferences. Lower alpha means more non-IID (more skewed).

    Args:
        ratings_df: DataFrame with ratings [user_id, movie_id, rating, timestamp]
        movies_df: DataFrame with movies [movie_id, title, genres]
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter (0.1-1.0, lower = more non-IID)
        min_ratings_per_client: Minimum number of ratings per client
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping client_id -> DataFrame with ratings for that client
    """
    np.random.seed(seed)

    # Get user genre preferences
    print("Computing user genre preferences...")
    user_genre_prefs = compute_user_genre_distribution(ratings_df, movies_df)
    all_users = user_genre_prefs.index.tolist()
    num_users = len(all_users)

    # Sample from Dirichlet distribution for each genre
    # This determines how genres are distributed across clients
    num_genres = user_genre_prefs.shape[1]
    print(f"Using Dirichlet(alpha={alpha}) to partition {num_users} users across {num_clients} clients...")

    # For each client, sample genre preference distribution
    client_genre_distributions = np.random.dirichlet([alpha] * num_genres, num_clients)

    # Assign users to clients based on genre similarity
    # Compute similarity between each user's genre preferences and client distributions
    user_to_client = {}
    users_remaining = set(all_users)

    # Convert user preferences to numpy array
    user_genre_matrix = user_genre_prefs.values

    # Assign users greedily to maximize genre alignment
    for user_id, user_prefs in zip(all_users, user_genre_matrix):
        # Compute KL divergence or cosine similarity with each client
        similarities = []
        for client_dist in client_genre_distributions:
            # Using negative KL divergence (higher is better)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            kl_div = -np.sum(
                user_prefs * np.log((user_prefs + epsilon) / (client_dist + epsilon))
            )
            similarities.append(kl_div)

        # Assign to client with highest similarity
        best_client = np.argmax(similarities)
        user_to_client[user_id] = best_client

    # Create partitions
    print("Creating client partitions...")
    client_partitions = {}
    for client_id in range(num_clients):
        # Get users assigned to this client
        client_users = [uid for uid, cid in user_to_client.items() if cid == client_id]

        # Get ratings for these users
        client_ratings = ratings_df[ratings_df["user_id"].isin(client_users)].copy()

        # Check minimum ratings constraint
        if len(client_ratings) < min_ratings_per_client:
            print(
                f"Warning: Client {client_id} has only {len(client_ratings)} ratings "
                f"(min: {min_ratings_per_client})"
            )

        client_partitions[client_id] = client_ratings

        print(
            f"Client {client_id}: {len(client_users)} users, "
            f"{len(client_ratings)} ratings"
        )

    # Print statistics
    print("\nPartitioning Statistics:")
    print(f"Total users: {num_users}")
    print(f"Total ratings: {len(ratings_df)}")
    print(
        f"Ratings per client: min={min(len(p) for p in client_partitions.values())}, "
        f"max={max(len(p) for p in client_partitions.values())}, "
        f"mean={np.mean([len(p) for p in client_partitions.values()]):.1f}"
    )

    return client_partitions


def create_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets.

    Args:
        ratings_df: DataFrame with ratings
        test_ratio: Fraction of ratings to use for testing
        seed: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(seed)

    # Shuffle and split
    shuffled = ratings_df.sample(frac=1, random_state=seed)
    split_idx = int(len(shuffled) * (1 - test_ratio))

    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")
    return train_df, test_df


def create_global_mappings(ratings_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create global user and item ID mappings.

    Args:
        ratings_df: DataFrame with all ratings

    Returns:
        Tuple of (user2idx, idx2user, item2idx, idx2item)
    """
    unique_users = sorted(ratings_df["user_id"].unique())
    unique_items = sorted(ratings_df["movie_id"].unique())

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}

    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx2item = {idx: iid for iid, idx in item2idx.items()}

    print(f"Created mappings: {len(user2idx)} users, {len(item2idx)} items")
    return user2idx, idx2user, item2idx, idx2item


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    data_dir: str = "./data",
):
    """
    Load and partition MovieLens 1M data for federated learning.

    Args:
        partition_id: ID of this client partition
        num_partitions: Total number of client partitions
        alpha: Dirichlet concentration parameter
        test_ratio: Ratio of test data
        batch_size: Batch size for DataLoader
        data_dir: Directory for data

    Returns:
        Tuple of (trainloader, testloader, num_users, num_items, user2idx, item2idx)
    """
    from torch.utils.data import DataLoader

    # Download and load data (only once)
    download_movielens_1m(data_dir)
    ratings_df, movies_df, _ = load_movielens_1m(data_dir)

    # Create global mappings
    user2idx, _, item2idx, _ = create_global_mappings(ratings_df)

    # Partition data using Dirichlet
    partitions = dirichlet_partition_users(
        ratings_df,
        movies_df,
        num_clients=num_partitions,
        alpha=alpha,
    )

    # Get this client's partition
    client_ratings = partitions[partition_id]

    # Split into train/test
    train_df, test_df = create_train_test_split(client_ratings, test_ratio=test_ratio)

    # Create datasets
    train_dataset = MovieLensDataset(train_df, user2idx, item2idx)
    test_dataset = MovieLensDataset(test_df, user2idx, item2idx)

    # Create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_users = len(user2idx)
    num_items = len(item2idx)

    return trainloader, testloader, num_users, num_items, user2idx, item2idx
