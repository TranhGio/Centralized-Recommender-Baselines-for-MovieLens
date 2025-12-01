"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization."""

import torch
import json
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from federated_baseline_cf.task import get_model, test, evaluate_ranking
from federated_baseline_cf.dataset import load_full_data

# Create ServerApp
app = ServerApp()

# Global control variate for SCAFFOLD (server state, persists across rounds)
_global_control_variate = None


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
                if f"precision@{k}" in metrics:
                    print(f"    Precision:  {metrics[f'precision@{k}']:.4f}")
                if f"recall@{k}" in metrics:
                    print(f"    Recall:     {metrics[f'recall@{k}']:.4f}")
                if f"f1@{k}" in metrics:
                    print(f"    F1:         {metrics[f'f1@{k}']:.4f}")
                if f"ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'ndcg@{k}']:.4f}")
                if f"map@{k}" in metrics:
                    print(f"    MAP:        {metrics[f'map@{k}']:.4f}")

            # Diversity/Popularity metrics (only for first K value to avoid repetition)
            k = sorted(k_values)[0]
            has_diversity = any(f"{m}@{k}" in metrics for m in ['coverage', 'novelty'])
            if has_diversity:
                print("\n📈 Diversity/Popularity Metrics:")
                for k in sorted(k_values):
                    print(f"\n  @ K={k}:")
                    if f"coverage@{k}" in metrics:
                        print(f"    Coverage:   {metrics[f'coverage@{k}']:.4f}")
                    if f"novelty@{k}" in metrics:
                        print(f"    Novelty:    {metrics[f'novelty@{k}']:.4f}")

    print(f"\n{'='*70}\n")


def run_scaffold_training(
    grid: Grid,
    context: Context,
    global_model: torch.nn.Module,
    num_rounds: int,
    fraction_train: float,
    lr: float,
) -> ArrayRecord:
    """
    Run SCAFFOLD federated learning with control variates.

    SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) uses
    control variates to correct for client drift in heterogeneous settings.

    Algorithm:
        Server maintains: global model x, global control variate c
        Each client i maintains: local control variate c_i

        For each round:
            1. Server sends (x, c) to selected clients
            2. Each client i:
                - Trains with corrected gradients: g_corrected = g - c_i + c
                - Computes delta_c_i = c_i_new - c_i (control variate update)
                - Sends (x_i, delta_c_i) to server
            3. Server:
                - Aggregates model: x_new = weighted_average(x_i)
                - Updates control variate: c_new = c + (1/N) * sum(delta_c_i)

    Args:
        grid: Flower Grid for client communication
        context: Flower Context with run configuration
        global_model: Initial global model
        num_rounds: Number of federated rounds
        fraction_train: Fraction of clients to train each round
        lr: Learning rate

    Returns:
        Final aggregated model weights as ArrayRecord
    """
    global _global_control_variate

    # Initialize global control variate to zeros (same shape as model parameters)
    if _global_control_variate is None:
        _global_control_variate = [
            torch.zeros_like(p).cpu() for p in global_model.parameters()
        ]

    # Convert global control variate to list format for transmission
    def control_variate_to_list(cv):
        return [c.tolist() for c in cv]

    # Current global model weights
    current_arrays = ArrayRecord(global_model.state_dict())

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- SCAFFOLD Round {round_num}/{num_rounds} ---")

        # Prepare train config with global control variate
        train_config = ConfigRecord({
            "lr": lr,
            "proximal_mu": 0.0,  # SCAFFOLD doesn't use proximal term
            "global_c": control_variate_to_list(_global_control_variate),
        })

        # Create content to send to clients
        content = RecordDict({
            "arrays": current_arrays,
            "config": train_config,
        })

        # Send train messages to clients and collect replies
        node_ids = list(grid.get_node_ids())
        num_clients_to_select = max(1, int(len(node_ids) * fraction_train))
        selected_nodes = node_ids[:num_clients_to_select]

        print(f"  Selected {len(selected_nodes)} clients for training")

        # Send messages to selected clients
        replies = []
        for node_id in selected_nodes:
            reply = grid.send_receive(
                node_id=node_id,
                content=content,
                message_type="train",
            )
            replies.append((node_id, reply))

        # Aggregate model weights (weighted average based on num-examples)
        total_examples = 0
        weighted_state_dict = None

        # Aggregate control variate updates
        aggregated_delta_c = [
            torch.zeros_like(c) for c in _global_control_variate
        ]
        num_scaffold_clients = 0

        for node_id, reply in replies:
            # Get number of examples for weighting
            num_examples = reply.content["metrics"].get("num-examples", 1)
            total_examples += num_examples

            # Get model state dict
            client_state_dict = reply.content["arrays"].to_torch_state_dict()

            # Initialize or accumulate weighted state dict
            if weighted_state_dict is None:
                weighted_state_dict = OrderedDict()
                for key, value in client_state_dict.items():
                    weighted_state_dict[key] = value.float() * num_examples
            else:
                for key, value in client_state_dict.items():
                    weighted_state_dict[key] += value.float() * num_examples

            # Aggregate SCAFFOLD control variate updates
            if "scaffold_data" in reply.content:
                delta_c_data = reply.content["scaffold_data"].get("delta_c", None)
                if delta_c_data is not None:
                    for i, dc in enumerate(delta_c_data):
                        aggregated_delta_c[i] += torch.tensor(dc)
                    num_scaffold_clients += 1

        # Compute weighted average of model weights
        if weighted_state_dict is not None and total_examples > 0:
            for key in weighted_state_dict:
                weighted_state_dict[key] /= total_examples
            current_arrays = ArrayRecord(weighted_state_dict)

        # Update global control variate: c_new = c + (1/N) * sum(delta_c_i)
        if num_scaffold_clients > 0:
            for i in range(len(_global_control_variate)):
                _global_control_variate[i] = (
                    _global_control_variate[i] +
                    aggregated_delta_c[i] / len(node_ids)  # Divide by total clients (N)
                )

        # Collect and print training metrics
        train_losses = []
        for node_id, reply in replies:
            train_loss = reply.content["metrics"].get("train_loss", 0.0)
            train_losses.append(train_loss)

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        print(f"  Average train loss: {avg_train_loss:.4f}")
        print(f"  Total examples: {total_examples}")

    print(f"\nSCAFFOLD training completed after {num_rounds} rounds")

    return current_arrays


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

    # FedProx configuration
    strategy_name: str = context.run_config.get("strategy", "fedavg").lower()
    proximal_mu: float = context.run_config.get("proximal-mu", 0.0)

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

    # Start Federated Learning
    print(f"\nStarting Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}%")
    print(f"  Ranking evaluation: {'Enabled' if context.run_config.get('enable-ranking-eval', True) else 'Disabled'}")
    if context.run_config.get('enable-ranking-eval', True):
        k_values_str = context.run_config.get('ranking-k-values', "5,10,20")
        print(f"  K values: {k_values_str}")

    # Initialize strategy based on configuration
    if strategy_name == "scaffold":
        # SCAFFOLD: Custom training loop with control variates
        print(f"  Strategy: SCAFFOLD (control variates for client drift correction)")

        result_arrays = run_scaffold_training(
            grid=grid,
            context=context,
            global_model=global_model,
            num_rounds=num_rounds,
            fraction_train=fraction_train,
            lr=lr,
        )

        # Create a result-like object for compatibility with rest of the code
        class ScaffoldResult:
            def __init__(self, arrays):
                self.arrays = arrays

        result = ScaffoldResult(result_arrays)

    elif strategy_name == "fedprox":
        # FedProx: Built-in Flower strategy with proximal term
        strategy = FedProx(
            fraction_train=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: FedProx (proximal_mu={proximal_mu})")

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "proximal_mu": proximal_mu}),
            num_rounds=num_rounds,
        )

    else:
        # FedAvg: Default Flower strategy
        strategy = FedAvg(
            fraction_train=fraction_train,
        )
        print(f"  Strategy: FedAvg")

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "proximal_mu": proximal_mu}),
            num_rounds=num_rounds,
        )

    # Print training complete message
    print("\n" + "="*70)
    print("FEDERATED TRAINING COMPLETE")
    print("="*70)
    print(f"Total rounds completed: {num_rounds}")
    print("="*70)

    # =========================================================================
    # CENTRALIZED EVALUATION: Run evaluation on server with final model
    # =========================================================================
    print("\n📊 Running centralized evaluation with final model...")

    # Load final model weights from result
    final_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    final_model.load_state_dict(result.arrays.to_torch_state_dict())

    # Auto-detect device (CUDA if available and compatible, else CPU)
    # Note: RTX 5090 (sm_120) requires PyTorch nightly with CUDA 12.8+
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a small tensor
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            device = torch.device("cuda:0")
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"  CUDA available but not compatible: {e}")
            print(f"  Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"  Using CPU")
    final_model.to(device)

    # Load full test data for evaluation
    trainloader, testloader, _, _, _, _ = load_full_data(
        test_ratio=0.2,
        batch_size=256,
    )

    # Compute rating prediction metrics (RMSE, MAE)
    print("  Computing rating prediction metrics...")
    eval_loss, rating_metrics = test(
        model=final_model,
        testloader=testloader,
        device=str(device),
        model_type=model_type,
    )

    # Compute ranking metrics (Hit Rate, Precision, Recall, F1, NDCG, MAP, Coverage, Novelty, MRR)
    print("  Computing ranking metrics...")
    k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
    k_values = [int(k.strip()) for k in k_values_str.split(",")]

    ranking_metrics = evaluate_ranking(
        model=final_model,
        testloader=testloader,
        device=str(device),
        k_values=k_values,
        trainloader=trainloader,  # For computing item popularity
    )

    # Combine all metrics
    final_metrics = {
        "eval_loss": float(eval_loss),
        **rating_metrics,
        **ranking_metrics,
    }

    # Print evaluation results
    print_evaluation_metrics(num_rounds, final_metrics, context)

    # Create results JSON structure similar to centralized results
    results_data = {
        "model_name": f"{model_type.upper()}_MF_Federated_{strategy_name.upper()}",
        "dataset": "ml-1m",
        "federated_config": {
            "num_rounds": num_rounds,
            "num_clients": 10,  # Adjust based on your config
            "fraction_train": fraction_train,
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "learning_rate": lr,
        },
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": num_rounds,
    }

    # Save results to JSON file
    print("\nSaving evaluation results...")
    results_dir = Path("../results/federated")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = results_dir / f"{model_type}_mf_{strategy_name}_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to: {results_filename.resolve()}")

    # Optionally save model weights (commented out by default)
    # print("\nSaving final model weights...")
    # state_dict = result.arrays.to_torch_state_dict()
    # model_filename = results_dir / f"final_model_{model_type}_d{embedding_dim}.pt"
    # torch.save(state_dict, model_filename)
    # print(f"Model saved to: {model_filename.resolve()}")
