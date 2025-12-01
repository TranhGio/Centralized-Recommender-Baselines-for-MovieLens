# Partition Visualizations for MovieLens 1M Dataset

This directory contains visualizations showing how the MovieLens 1M dataset is partitioned across federated clients using the **Dirichlet distribution** for creating non-IID (non-independent and identically distributed) data.

## Understanding the Dirichlet Parameter (α)

The Dirichlet concentration parameter `α` controls the degree of non-IID distribution:

- **α = 0.1**: **Highly non-IID** (very skewed distribution)
  - Most data concentrated in a few clients
  - Clients have very different genre preferences
  - Mimics extreme real-world heterogeneity

- **α = 0.5**: **Moderately non-IID** (recommended for experiments)
  - Balanced between heterogeneity and fairness
  - Realistic federated learning scenario

- **α = 1.0**: **Less non-IID** (more balanced)
  - More uniform distribution across clients
  - Closer to IID but still maintains some heterogeneity

## Generated Visualizations

### 1. Partition Sizes (`partition_sizes_alpha_*.png`)
Shows the distribution of:
- **Ratings per client**: How many ratings each client has
- **Users per client**: Number of unique users on each client
- **Movies per client**: Number of unique movies rated by each client

**What to look for:**
- Large variation = high non-IID
- Smaller bars indicate clients with less data (challenging for training)

### 2. Genre Distribution Heatmap (`genre_distribution_alpha_*.png`)
**Heatmap showing genre proportions** for each client across all 18 MovieLens genres.

**What to look for:**
- **Bright colors** indicate high proportion of that genre
- **Different patterns per client** = successful non-IID partitioning
- Clients should specialize in certain genres (e.g., Client 0 loves Horror, Client 8 prefers Action)

**Key insight:** This visualization proves the non-IID nature - each client has different genre preferences!

### 3. Rating Distribution (`rating_distribution_alpha_*.png`)
Shows the **distribution of rating values (1-5 stars)** for each client.

**What to look for:**
- Most ratings should be 3-5 (typical for MovieLens)
- Variation in distribution patterns across clients
- Some clients may be more critical (lower ratings) or more generous (higher ratings)

### 4. User Activity Distribution (`user_activity_alpha_*.png`)
**Histogram of ratings per user** within each client.

**What to look for:**
- Shows how active users are on each client
- Mean line (red dashed) shows average activity
- Some clients have power users (many ratings), others have casual users

### 5. Alpha Comparison (`alpha_comparison.png`)
**Side-by-side comparison** of different α values.

**Left plot:** Line graph showing ratings distribution across clients
- **Log scale** to handle large variations
- Lower α = steeper curves (more skewed)

**Right plot:** Box plot showing spread
- **Larger boxes** = more heterogeneous distribution
- **Coefficient of Variation:**
  - α=0.1: CV=2.97 (most heterogeneous)
  - α=0.5: CV=2.03 (moderate)
  - α=1.0: CV=1.74 (least heterogeneous)

### 6. Partition Summary CSVs (`partition_summary_alpha_*.csv`)
**Detailed statistics** for each client partition:
- Number of ratings, users, movies
- Average rating
- Ratings per user
- Top 3 genres with counts

**Use this for:**
- Quick reference during experiments
- Identifying which clients have which characteristics
- Understanding client specialization

## Key Observations

### Alpha = 0.1 (Highly Non-IID)
```
Client 8: 943,993 ratings (89% of all data!)
Client 4: 0 ratings (empty client - too extreme)
CV: 2.97 (very high heterogeneity)
```
- ⚠️ **Too extreme** - one client dominates, some clients are empty
- Not recommended for experiments

### Alpha = 0.5 (Recommended)
```
Client 2: 619,815 ratings (largest)
Client 0: 990 ratings (smallest)
CV: 2.03 (moderate heterogeneity)
```
- ✅ **Best balance** - realistic non-IID without extreme cases
- All clients have sufficient data for training
- Recommended for federated learning experiments

### Alpha = 1.0 (Less Non-IID)
```
Client 9: 499,994 ratings (largest)
Client 4: 192 ratings (smallest)
CV: 1.74 (lower heterogeneity)
```
- More balanced but still maintains heterogeneity
- Good for comparing with more IID scenarios

## Genre Specialization Examples (α=0.5)

Based on the partition summary:

- **Client 0**: Horror specialist (484/990 ratings = 49%)
- **Client 1**: Drama lover (7,199/8,722 ratings = 83%)
- **Client 2**: Balanced mainstream (Drama, Comedy, Action)
- **Client 3**: Children's content (1,732 ratings)
- **Client 7**: Sci-Fi enthusiast (12,296 ratings)

This specialization is exactly what we want for federated learning research - it mimics real-world scenarios where different users/regions have different preferences!

## How to Regenerate

Run the visualization script:
```bash
python visualize_partitions.py
```

This will regenerate all visualizations for α ∈ {0.1, 0.5, 1.0}.

## Understanding Non-IID in Federated Learning

**Why non-IID matters:**
1. **Realistic**: Real-world federated learning scenarios have heterogeneous data
2. **Challenging**: Non-IID data makes model aggregation harder
3. **Research value**: Tests if your FL algorithm works in practical settings

**What this partitioning achieves:**
- Users with similar genre preferences are grouped together
- Each client becomes a "community" with shared interests
- Mimics geographical or demographic user clustering
- Provides a challenging but realistic FL benchmark

## Citation

If you use these visualizations or the Dirichlet partitioning approach in your research, please cite:

```
MovieLens 1M Dataset:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS)
5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
```
