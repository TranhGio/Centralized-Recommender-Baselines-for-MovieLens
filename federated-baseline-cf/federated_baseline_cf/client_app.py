"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_baseline_cf.task import get_model, load_data
from federated_baseline_cf.task import test as test_fn
from federated_baseline_cf.task import train as train_fn
from federated_baseline_cf.task import train_scaffold as train_scaffold_fn
from federated_baseline_cf.task import evaluate_ranking

# Flower ClientApp
app = ClientApp()

# Cache for device detection (avoid repeated CUDA tests)
_device_cache = None

# Cache for SCAFFOLD control variates (persists across rounds per client)
# Key: partition_id, Value: list of tensors (local control variate c_i)
_scaffold_control_variates = {}


def get_device():
    """Get device with safe CUDA detection (handles incompatible GPU architectures)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a small tensor
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            # CUDA available but not compatible (e.g., RTX 5090 with old PyTorch)
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")

    return _device_cache


@app.train()
def train(msg: Message, context: Context):
    """Train the Matrix Factorization model on local data."""

    # Get model configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = context.run_config.get("embedding-dim", 64)
    dropout = context.run_config.get("dropout", 0.1)

    # Load the model and initialize it with the received weights
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Use safe device detection (handles incompatible GPU architectures)
    device = get_device()
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config.get("alpha", 0.5)
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
    )

    # Get strategy from config
    strategy = context.run_config.get("strategy", "fedavg").lower()

    # === SCAFFOLD Training ===
    if strategy == "scaffold":
        global _scaffold_control_variates

        # Get global control variate from server (list of tensors as lists)
        global_c_data = msg.content["config"].get("global_c", None)

        # Convert global_c from list format back to tensors
        if global_c_data is not None:
            global_c = [torch.tensor(c, device=device) for c in global_c_data]
        else:
            # Initialize to zeros if not received (first round)
            global_c = [torch.zeros_like(p, device=device) for p in model.parameters()]

        # Get or initialize local control variate from cache
        if partition_id not in _scaffold_control_variates:
            _scaffold_control_variates[partition_id] = [
                torch.zeros_like(p, device=device) for p in model.parameters()
            ]
        local_c = _scaffold_control_variates[partition_id]

        # Train with SCAFFOLD correction
        train_loss, delta_c = train_scaffold_fn(
            model=model,
            trainloader=trainloader,
            epochs=context.run_config["local-epochs"],
            lr=msg.content["config"]["lr"],
            device=device,
            model_type=model_type,
            weight_decay=context.run_config.get("weight-decay", 1e-5),
            num_negatives=context.run_config.get("num-negatives", 1),
            global_c=global_c,
            local_c=local_c,
        )

        # Update local control variate: c_i_new = c_i + delta_c
        _scaffold_control_variates[partition_id] = [
            lc + dc for lc, dc in zip(local_c, delta_c)
        ]

        # Construct reply with model weights AND delta_c for server aggregation
        model_record = ArrayRecord(model.state_dict())

        # Convert delta_c tensors to lists for transmission
        delta_c_data = [dc.cpu().tolist() for dc in delta_c]

        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)

        # Include delta_c in config for server to aggregate
        delta_c_record = {"delta_c": delta_c_data}

        content = RecordDict({
            "arrays": model_record,
            "metrics": metric_record,
            "scaffold_data": delta_c_record,
        })
        return Message(content=content, reply_to=msg)

    # === FedAvg / FedProx Training ===
    else:
        # Get proximal_mu from config (0.0 means standard FedAvg behavior)
        proximal_mu = msg.content["config"].get("proximal_mu", 0.0)

        # Save global parameters for proximal term (only if proximal_mu > 0)
        global_params = None
        if proximal_mu > 0:
            global_params = [p.detach().clone() for p in model.parameters()]

        # Call the training function
        train_loss = train_fn(
            model=model,
            trainloader=trainloader,
            epochs=context.run_config["local-epochs"],
            lr=msg.content["config"]["lr"],
            device=device,
            model_type=model_type,
            weight_decay=context.run_config.get("weight-decay", 1e-5),
            num_negatives=context.run_config.get("num-negatives", 1),
            # FedProx parameters
            proximal_mu=proximal_mu,
            global_params=global_params,
        )

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the Matrix Factorization model on local data."""

    # Get model configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = context.run_config.get("embedding-dim", 64)
    dropout = context.run_config.get("dropout", 0.1)

    # Load the model and initialize it with the received weights
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Use safe device detection (handles incompatible GPU architectures)
    device = get_device()
    model.to(device)

    # Load the data (both train and test for item popularity computation)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config.get("alpha", 0.5)
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
    )

    # Call the evaluation function (rating prediction metrics)
    eval_loss, metrics = test_fn(
        model=model,
        testloader=testloader,
        device=device,
        model_type=model_type,
    )

    # Construct result metrics
    result_metrics = {
        "eval_loss": eval_loss,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "num-examples": len(testloader.dataset),
    }

    # Add ranking metrics if enabled
    enable_ranking_eval = context.run_config.get("enable-ranking-eval", True)
    if enable_ranking_eval:
        # Get K values from config (parse comma-separated string)
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        # Compute ranking metrics (pass trainloader for item popularity computation)
        ranking_metrics = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=device,
            k_values=k_values,
            trainloader=trainloader,
        )

        # Add ranking metrics to results
        result_metrics.update(ranking_metrics)

    metric_record = MetricRecord(result_metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
