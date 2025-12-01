# Federated Baseline Collaborative Filtering

A Flower federated learning project for collaborative filtering on the MovieLens 1M dataset, implementing both **Matrix Factorization (MF)** and **Neural Collaborative Filtering (NCF)** with **Dirichlet partitioning** for non-IID data distribution.

## Features

- ✅ **MovieLens 1M Dataset**: Automatic download and preprocessing
- ✅ **Dirichlet Partitioning**: Creates realistic non-IID data distribution based on genre preferences
- ✅ **Two Model Approaches**:
  - Matrix Factorization (MF) - PyTorch implementation (TODO)
  - Neural Collaborative Filtering (NCF) (TODO)
- ✅ **Comprehensive Visualizations**: Analyze data partitioning across clients
- 🔄 **Evaluation Metrics**: RMSE, MAE, Precision@K, Recall@K, NDCG (TODO)
- 🔄 **Federated Training**: Client-server architecture with FedAvg aggregation (TODO)

## Dataset

**MovieLens 1M**:
- 1,000,209 ratings from 6,040 users on 3,883 movies
- Rating scale: 1-5 stars
- 18 movie genres
- Automatically downloaded on first run

## Quick Start

### Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

### Test Dataset Loading and Partitioning

Test the MovieLens 1M dataset loading and Dirichlet partitioning:

```bash
python test_dataset.py
```

This will:
1. Download MovieLens 1M dataset (if not already downloaded)
2. Test data loading
3. Test Dirichlet partitioning with different α values (0.1, 0.5, 1.0)
4. Test DataLoader creation

### Visualize Data Partitions

Generate comprehensive visualizations of data distribution:

```bash
python visualize_partitions.py
```

This creates visualizations in the `./figures/` directory:
- **Partition sizes**: Ratings, users, and movies per client
- **Genre distribution heatmap**: Shows non-IID characteristics
- **Rating distribution**: Rating values (1-5) per client
- **User activity**: Ratings per user distribution
- **Alpha comparison**: Compare different Dirichlet parameters
- **Summary CSVs**: Detailed statistics for each partition

See [`figures/README.md`](./figures/README.md) for detailed explanations of each visualization.

### Understanding the Dirichlet Parameter (α)

- **α = 0.1**: Highly non-IID (very skewed, some clients may be empty)
- **α = 0.5**: Moderately non-IID (**recommended** for experiments)
- **α = 1.0**: Less non-IID (more balanced)

Lower α means more heterogeneous data distribution across clients.

## Run with the Simulation Engine (TODO)

In the `federated-baseline-cf` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Project Structure

```
federated-baseline-cf/
├── federated_baseline_cf/
│   ├── __init__.py
│   ├── dataset.py           # ✅ MovieLens 1M loader & Dirichlet partitioner
│   ├── task.py              # 🔄 Models & training logic (TODO)
│   ├── client_app.py        # 🔄 Flower client (TODO: update for CF)
│   └── server_app.py        # 🔄 Flower server (TODO: update for CF)
├── data/
│   └── ml-1m/              # MovieLens 1M dataset (auto-downloaded)
├── figures/                 # ✅ Generated visualizations
│   ├── README.md           # Detailed explanation of visualizations
│   ├── partition_sizes_alpha_*.png
│   ├── genre_distribution_alpha_*.png
│   ├── rating_distribution_alpha_*.png
│   ├── user_activity_alpha_*.png
│   ├── alpha_comparison.png
│   └── partition_summary_alpha_*.csv
├── test_dataset.py         # ✅ Test dataset loading
├── visualize_partitions.py # ✅ Generate visualizations
├── pyproject.toml          # Project config & dependencies
└── README.md               # This file
```

## Implementation Details

### Dirichlet Partitioning Algorithm

The partitioning creates non-IID data by:
1. Computing each user's genre preference distribution
2. Sampling client genre distributions from Dirichlet(α)
3. Assigning users to clients based on genre similarity (KL divergence)
4. Each client receives all ratings from its assigned users

**Result**: Clients have users with similar genre preferences, creating realistic heterogeneity.

### Example Partition Statistics (α=0.5)

```
Client 0:    990 ratings,    8 users (Horror specialist)
Client 1:  8,722 ratings,  108 users (Drama lover)
Client 2: 619,815 ratings, 3,427 users (Mainstream - largest)
...
Client 9:  5,899 ratings,   60 users (Mixed preferences)
```

**Coefficient of Variation**: 2.03 (indicates strong heterogeneity)

## Next Steps

- [ ] Implement Matrix Factorization model (PyTorch)
- [ ] Implement Neural Collaborative Filtering model
- [ ] Update `task.py` with training/evaluation logic
- [ ] Update client/server apps for recommendation models
- [ ] Implement evaluation metrics (RMSE, MAE, P@K, R@K, NDCG)
- [ ] Run federated training experiments
- [ ] Compare centralized vs federated performance

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

## Citation

If you use this code or the MovieLens dataset, please cite:

```bibtex
@article{harper2015movielens,
  title={The MovieLens datasets: History and context},
  author={Harper, F. Maxwell and Konstan, Joseph A},
  journal={ACM Transactions on Interactive Intelligent Systems (TiiS)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  publisher={ACM New York, NY, USA}
}
```
