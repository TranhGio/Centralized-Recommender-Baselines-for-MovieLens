"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization."""

import torch
import json
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated_baseline_cf.task import get_model

# Create ServerApp
app = ServerApp()


def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Aggregate evaluation metrics from multiple clients using weighted average.

    NOTE: This function is available for custom metric aggregation but is not
    currently used. Flower's new ServerApp API handles metric aggregation
    automatically based on num-examples.

    This function aggregates both rating prediction metrics (RMSE, MAE) and
    ranking metrics (Hit Rate, Precision, Recall, NDCG, MRR) across clients.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client

    Returns:
        Dictionary of aggregated metrics
    """
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Aggregate metrics using weighted average
    aggregated = {}

    # Get all metric keys from first client (assumes all clients report same metrics)
    if metrics:
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "num-examples":
                continue

            # Weighted average: sum(metric * num_examples) / total_examples
            weighted_sum = sum(
                metrics_dict.get(key, 0.0) * num_examples
                for num_examples, metrics_dict in metrics
            )
            aggregated[key] = weighted_sum / total_examples

    return aggregated


def print_evaluation_metrics(round_num: int, metrics: Dict[str, float], context: Context):
    """
    Pretty print evaluation metrics for a federated round.

    Args:
        round_num: Current federated learning round
        metrics: Aggregated metrics dictionary
        context: Flower context with configuration
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Results - Round {round_num}")
    print(f"{'='*70}")

    # Rating prediction metrics
    if "rmse" in metrics or "mae" in metrics:
        print("\n📊 Rating Prediction Metrics:")
        if "eval_loss" in metrics:
            print(f"  Loss:      {metrics['eval_loss']:.4f}")
        if "rmse" in metrics:
            print(f"  RMSE:      {metrics['rmse']:.4f}")
        if "mae" in metrics:
            print(f"  MAE:       {metrics['mae']:.4f}")

    # Ranking metrics
    enable_ranking = context.run_config.get("enable-ranking-eval", True)
    if enable_ranking:
        # Parse K values from comma-separated string
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        # Check if we have any ranking metrics
        has_ranking = any(f"hit_rate@{k}" in metrics for k in k_values)

        if has_ranking:
            print("\n🎯 Ranking Metrics:")

            # MRR (not K-dependent)
            if "mrr" in metrics:
                print(f"  MRR:       {metrics['mrr']:.4f}")

            # Metrics for each K value
            for k in sorted(k_values):
                print(f"\n  @ K={k}:")
                if f"hit_rate@{k}" in metrics:
                    print(f"    Hit Rate:   {metrics[f'hit_rate@{k}']:.4f}")
                if f"accuracy@{k}" in metrics:
                    print(f"    Accuracy:   {metrics[f'accuracy@{k}']:.4f}")
                if f"precision@{k}" in metrics:
                    print(f"    Precision:  {metrics[f'precision@{k}']:.4f}")
                if f"recall@{k}" in metrics:
                    print(f"    Recall:     {metrics[f'recall@{k}']:.4f}")
                if f"ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'ndcg@{k}']:.4f}")

    print(f"\n{'='*70}\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    model_type: str = context.run_config.get("model-type", "bpr")
    embedding_dim: int = context.run_config.get("embedding-dim", 64)
    dropout: float = context.run_config.get("dropout", 0.1)

    # Load global Matrix Factorization model
    print(f"\nInitializing {model_type.upper()} Matrix Factorization model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout: {dropout}")

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters: {num_params:,}")

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    # Note: Flower automatically does weighted averaging of metrics based on num-examples
    strategy = FedAvg(
        fraction_train=fraction_train,
    )

    # Start strategy, run FedAvg for `num_rounds`
    print(f"\nStarting Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}%")
    print(f"  Ranking evaluation: {'Enabled' if context.run_config.get('enable-ranking-eval', True) else 'Disabled'}")
    if context.run_config.get('enable-ranking-eval', True):
        k_values_str = context.run_config.get('ranking-k-values', "5,10,20")
        print(f"  K values: {k_values_str}")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Print final evaluation metrics summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total rounds completed: {num_rounds}")
    print("Note: Evaluation metrics are displayed during training.")
    print("="*70)

    # Extract final metrics from the last round
    final_metrics = {}
    if hasattr(result, 'metrics_distributed') and result.metrics_distributed:
        # Get metrics from the last round
        last_round = max(result.metrics_distributed.keys())
        last_round_metrics = result.metrics_distributed[last_round]

        # Extract the aggregated metrics
        if last_round_metrics:
            for key, value in last_round_metrics.items():
                if key != "num-examples":
                    final_metrics[key] = float(value)

    # Create results JSON structure similar to centralized results
    results_data = {
        "model_name": f"{model_type.upper()}_MF_Federated",
        "dataset": "ml-1m",
        "federated_config": {
            "num_rounds": num_rounds,
            "num_clients": 10,  # Adjust based on your config
            "fraction_train": fraction_train,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "learning_rate": lr,
        },
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": num_rounds,
    }

    # Add per-round metrics if available
    if hasattr(result, 'metrics_distributed') and result.metrics_distributed:
        results_data["metrics_per_round"] = {}
        for round_num, metrics in result.metrics_distributed.items():
            results_data["metrics_per_round"][str(round_num)] = {
                k: float(v) for k, v in metrics.items() if k != "num-examples"
            }

    # Save results to JSON file
    print("\nSaving evaluation results...")
    results_dir = Path("../results/federated")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = results_dir / f"{model_type}_mf_federated_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to: {results_filename.resolve()}")

    # Optionally save model weights (commented out by default)
    # print("\nSaving final model weights...")
    # state_dict = result.arrays.to_torch_state_dict()
    # model_filename = results_dir / f"final_model_{model_type}_d{embedding_dim}.pt"
    # torch.save(state_dict, model_filename)
    # print(f"Model saved to: {model_filename.resolve()}")
