"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_baseline_cf.task import get_model, load_data
from federated_baseline_cf.task import test as test_fn
from federated_baseline_cf.task import train as train_fn
from federated_baseline_cf.task import evaluate_ranking

# Flower ClientApp
app = ClientApp()


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print device info for verification
    print(f"🎮 Training on device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config.get("alpha", 0.5)
    _, testloader = load_data(
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

        # Compute ranking metrics
        ranking_metrics = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=device,
            k_values=k_values,
        )

        # Add ranking metrics to results
        result_metrics.update(ranking_metrics)

    metric_record = MetricRecord(result_metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
