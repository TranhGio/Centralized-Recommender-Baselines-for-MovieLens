"""Visualize MovieLens 1M data partitioning across federated clients."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from federated_baseline_cf.dataset import (
    download_movielens_1m,
    load_movielens_1m,
    dirichlet_partition_users,
    compute_user_genre_distribution,
)


def plot_partition_sizes(partitions, alpha, save_dir="./figures"):
    """Plot the distribution of ratings and users across clients."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_clients = len(partitions)
    client_ids = list(range(num_clients))

    ratings_per_client = [len(partitions[i]) for i in client_ids]
    users_per_client = [partitions[i]["user_id"].nunique() for i in client_ids]
    items_per_client = [partitions[i]["movie_id"].nunique() for i in client_ids]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Ratings per client
    axes[0].bar(client_ids, ratings_per_client, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Client ID', fontsize=12)
    axes[0].set_ylabel('Number of Ratings', fontsize=12)
    axes[0].set_title(f'Ratings Distribution (α={alpha})', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(ratings_per_client):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Users per client
    axes[1].bar(client_ids, users_per_client, color='coral', alpha=0.7)
    axes[1].set_xlabel('Client ID', fontsize=12)
    axes[1].set_ylabel('Number of Users', fontsize=12)
    axes[1].set_title(f'Users Distribution (α={alpha})', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(users_per_client):
        axes[1].text(i, v, str(v), ha='center', va='bottom', fontsize=9)

    # Plot 3: Items per client
    axes[2].bar(client_ids, items_per_client, color='mediumseagreen', alpha=0.7)
    axes[2].set_xlabel('Client ID', fontsize=12)
    axes[2].set_ylabel('Number of Movies', fontsize=12)
    axes[2].set_title(f'Movies Distribution (α={alpha})', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    for i, v in enumerate(items_per_client):
        axes[2].text(i, v, str(v), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/partition_sizes_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/partition_sizes_alpha_{alpha}.png")
    plt.close()


def plot_genre_distribution(partitions, movies_df, alpha, save_dir="./figures"):
    """Plot genre distribution for each client to show non-IID characteristics."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get all unique genres
    all_genres = set()
    for genres_str in movies_df["genres"].unique():
        all_genres.update(genres_str.split("|"))
    all_genres = sorted(all_genres)

    # Compute genre distribution for each client
    num_clients = len(partitions)
    genre_matrix = np.zeros((num_clients, len(all_genres)))

    for client_id in range(num_clients):
        client_ratings = partitions[client_id]
        if len(client_ratings) == 0:
            continue

        # Merge with movies to get genres
        merged = client_ratings.merge(movies_df[["movie_id", "genres"]], on="movie_id")

        # Count genre occurrences
        genre_counts = {genre: 0 for genre in all_genres}
        for genres_str in merged["genres"]:
            for genre in genres_str.split("|"):
                genre_counts[genre] += 1

        # Normalize to proportions
        total = sum(genre_counts.values())
        if total > 0:
            for i, genre in enumerate(all_genres):
                genre_matrix[client_id, i] = genre_counts[genre] / total

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        genre_matrix.T,
        xticklabels=[f'Client {i}' for i in range(num_clients)],
        yticklabels=all_genres,
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f'Genre Distribution Across Clients (α={alpha})\nShows Non-IID Data Distribution',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/genre_distribution_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/genre_distribution_alpha_{alpha}.png")
    plt.close()


def plot_rating_distribution(partitions, alpha, save_dir="./figures"):
    """Plot rating value distribution for each client."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_clients = len(partitions)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for client_id in range(num_clients):
        client_ratings = partitions[client_id]

        if len(client_ratings) > 0:
            rating_counts = client_ratings["rating"].value_counts().sort_index()

            axes[client_id].bar(rating_counts.index, rating_counts.values,
                               color='skyblue', edgecolor='navy', alpha=0.7)
            axes[client_id].set_title(f'Client {client_id}\n({len(client_ratings):,} ratings)',
                                     fontsize=10, fontweight='bold')
            axes[client_id].set_xlabel('Rating', fontsize=9)
            axes[client_id].set_ylabel('Count', fontsize=9)
            axes[client_id].set_xticks([1, 2, 3, 4, 5])
            axes[client_id].grid(axis='y', alpha=0.3)
        else:
            axes[client_id].text(0.5, 0.5, 'No Data', ha='center', va='center',
                                transform=axes[client_id].transAxes, fontsize=12)
            axes[client_id].set_title(f'Client {client_id}\n(0 ratings)',
                                     fontsize=10, fontweight='bold')

    fig.suptitle(f'Rating Distribution Per Client (α={alpha})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rating_distribution_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/rating_distribution_alpha_{alpha}.png")
    plt.close()


def plot_alpha_comparison(ratings_df, movies_df, alphas=[0.1, 0.5, 1.0],
                         num_clients=10, save_dir="./figures"):
    """Compare partitioning across different alpha values."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_stats = []

    for alpha in alphas:
        print(f"\nPartitioning with alpha={alpha}...")
        partitions = dirichlet_partition_users(
            ratings_df, movies_df, num_clients=num_clients, alpha=alpha, seed=42
        )

        ratings_per_client = [len(partitions[i]) for i in range(num_clients)]
        users_per_client = [partitions[i]["user_id"].nunique() for i in range(num_clients)]

        for client_id in range(num_clients):
            all_stats.append({
                'alpha': alpha,
                'client_id': client_id,
                'ratings': ratings_per_client[client_id],
                'users': users_per_client[client_id],
            })

    stats_df = pd.DataFrame(all_stats)

    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Ratings distribution comparison
    for alpha in alphas:
        data = stats_df[stats_df['alpha'] == alpha]
        axes[0].plot(data['client_id'], data['ratings'],
                    marker='o', label=f'α={alpha}', linewidth=2, markersize=8)

    axes[0].set_xlabel('Client ID', fontsize=12)
    axes[0].set_ylabel('Number of Ratings', fontsize=12)
    axes[0].set_title('Impact of α on Data Distribution\n(Lower α = More Non-IID)',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Plot 2: Box plot showing spread
    ratings_by_alpha = [stats_df[stats_df['alpha'] == a]['ratings'].values for a in alphas]
    bp = axes[1].boxplot(ratings_by_alpha, labels=[f'α={a}' for a in alphas],
                         patch_artist=True, showmeans=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[1].set_xlabel('Dirichlet Parameter (α)', fontsize=12)
    axes[1].set_ylabel('Ratings per Client', fontsize=12)
    axes[1].set_title('Distribution Spread Across α Values\n(Larger spread = More heterogeneous)',
                     fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/alpha_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/alpha_comparison.png")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("Alpha Comparison Statistics")
    print("="*80)
    for alpha in alphas:
        data = stats_df[stats_df['alpha'] == alpha]['ratings']
        print(f"\nα = {alpha}:")
        print(f"  Min ratings: {data.min():,}")
        print(f"  Max ratings: {data.max():,}")
        print(f"  Mean ratings: {data.mean():.1f}")
        print(f"  Std ratings: {data.std():.1f}")
        print(f"  Coefficient of Variation: {(data.std() / data.mean()):.2f}")


def plot_user_activity_distribution(partitions, alpha, save_dir="./figures"):
    """Plot user activity (ratings per user) distribution for each client."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_clients = len(partitions)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for client_id in range(num_clients):
        client_ratings = partitions[client_id]

        if len(client_ratings) > 0:
            user_activity = client_ratings.groupby('user_id').size()

            axes[client_id].hist(user_activity.values, bins=30,
                               color='mediumpurple', edgecolor='darkviolet', alpha=0.7)
            axes[client_id].set_title(f'Client {client_id}\n({len(user_activity)} users)',
                                     fontsize=10, fontweight='bold')
            axes[client_id].set_xlabel('Ratings per User', fontsize=9)
            axes[client_id].set_ylabel('Number of Users', fontsize=9)
            axes[client_id].grid(axis='y', alpha=0.3)

            # Add mean line
            mean_activity = user_activity.mean()
            axes[client_id].axvline(mean_activity, color='red', linestyle='--',
                                   linewidth=2, label=f'Mean: {mean_activity:.1f}')
            axes[client_id].legend(fontsize=8)
        else:
            axes[client_id].text(0.5, 0.5, 'No Data', ha='center', va='center',
                                transform=axes[client_id].transAxes, fontsize=12)
            axes[client_id].set_title(f'Client {client_id}\n(0 users)',
                                     fontsize=10, fontweight='bold')

    fig.suptitle(f'User Activity Distribution Per Client (α={alpha})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/user_activity_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/user_activity_alpha_{alpha}.png")
    plt.close()


def create_summary_statistics(partitions, movies_df, alpha, save_dir="./figures"):
    """Create a summary table of partition statistics."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_clients = len(partitions)
    summary_data = []

    for client_id in range(num_clients):
        client_ratings = partitions[client_id]

        if len(client_ratings) > 0:
            num_ratings = len(client_ratings)
            num_users = client_ratings["user_id"].nunique()
            num_movies = client_ratings["movie_id"].nunique()
            avg_rating = client_ratings["rating"].mean()

            # Get top genres
            merged = client_ratings.merge(movies_df[["movie_id", "genres"]], on="movie_id")
            genre_counts = {}
            for genres_str in merged["genres"]:
                for genre in genres_str.split("|"):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_genres_str = ", ".join([f"{g[0]} ({g[1]})" for g in top_genres])

            summary_data.append({
                'Client': client_id,
                'Ratings': num_ratings,
                'Users': num_users,
                'Movies': num_movies,
                'Avg Rating': f"{avg_rating:.2f}",
                'Ratings/User': f"{num_ratings/num_users:.1f}",
                'Top Genres': top_genres_str
            })
        else:
            summary_data.append({
                'Client': client_id,
                'Ratings': 0,
                'Users': 0,
                'Movies': 0,
                'Avg Rating': 'N/A',
                'Ratings/User': 'N/A',
                'Top Genres': 'N/A'
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_df.to_csv(f'{save_dir}/partition_summary_alpha_{alpha}.csv', index=False)
    print(f"Saved: {save_dir}/partition_summary_alpha_{alpha}.csv")

    # Print to console
    print(f"\n{'='*100}")
    print(f"Partition Summary (α={alpha})")
    print('='*100)
    print(summary_df.to_string(index=False))
    print('='*100)

    return summary_df


def visualize_all(alpha=0.5, num_clients=10, save_dir="./figures"):
    """Generate all visualizations for a given alpha value."""
    print(f"\n{'='*80}")
    print(f"Generating Visualizations for α={alpha}")
    print('='*80)

    # Load data
    download_movielens_1m("./data")
    ratings_df, movies_df, _ = load_movielens_1m("./data")

    # Create partitions
    print(f"\nCreating partitions with {num_clients} clients...")
    partitions = dirichlet_partition_users(
        ratings_df, movies_df, num_clients=num_clients, alpha=alpha, seed=42
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_partition_sizes(partitions, alpha, save_dir)
    plot_genre_distribution(partitions, movies_df, alpha, save_dir)
    plot_rating_distribution(partitions, alpha, save_dir)
    plot_user_activity_distribution(partitions, alpha, save_dir)
    create_summary_statistics(partitions, movies_df, alpha, save_dir)

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {save_dir}/")
    print('='*80)


if __name__ == "__main__":
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Generate visualizations for different alpha values
    for alpha in [0.1, 0.5, 1.0]:
        visualize_all(alpha=alpha, num_clients=10, save_dir="./figures")

    # Generate alpha comparison
    print("\nGenerating alpha comparison...")
    ratings_df, movies_df, _ = load_movielens_1m("./data")
    plot_alpha_comparison(ratings_df, movies_df, alphas=[0.1, 0.5, 1.0],
                         num_clients=10, save_dir="./figures")

    print("\n" + "="*80)
    print("All visualizations completed!")
    print(f"Check the ./figures/ directory for all generated plots")
    print("="*80)
