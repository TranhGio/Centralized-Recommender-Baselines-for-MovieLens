# Federated Learning Results Format

## Output File Location
```
results/federated/{model_type}_mf_federated_results.json
```

Examples:
- `results/federated/basic_mf_federated_results.json`
- `results/federated/bpr_mf_federated_results.json`

## JSON Structure

The results JSON will be saved in a format similar to your centralized results:

```json
{
    "model_name": "BPR_MF_Federated",
    "dataset": "ml-1m",
    "federated_config": {
        "num_rounds": 10,
        "num_clients": 10,
        "fraction_train": 1.0,
        "model_type": "bpr",
        "embedding_dim": 64,
        "dropout": 0.1,
        "learning_rate": 0.001
    },
    "timestamp": "2025-11-23T23:15:30.123456",
    "final_metrics": {
        "eval_loss": 0.8465727297839,
        "rmse": 0.9157275694647562,
        "mae": 0.7319138576648719,
        "hit_rate@5": 0.2642905830753715,
        "accuracy@5": 0.2642905830753715,
        "precision@5": 0.06552892844359283,
        "recall@5": 0.013413907720414038,
        "ndcg@5": 0.06792594958412523,
        "hit_rate@10": 0.43295973377864583,
        "accuracy@10": 0.43295973377864583,
        "precision@10": 0.06453760971766265,
        "recall@10": 0.02802978860670876,
        "ndcg@10": 0.06827944976632473,
        "hit_rate@20": 0.5762039312697435,
        "accuracy@20": 0.5762039312697435,
        "precision@20": 0.05636067135514205,
        "recall@20": 0.047462941909021235,
        "ndcg@20": 0.06813711571325785,
        "mrr": 0.17107744076118092
    },
    "training_rounds": 10,
    "metrics_per_round": {
        "1": {
            "eval_loss": 2.5261,
            "rmse": 1.4661,
            "mae": 1.2733,
            "accuracy@5": 0.27130,
            "hit_rate@5": 0.27130,
            ...
        },
        "2": {
            "eval_loss": 1.2233,
            "rmse": 1.0852,
            "mae": 0.9111,
            ...
        },
        ...
        "10": {
            "eval_loss": 0.8465,
            "rmse": 0.9157,
            "mae": 0.7319,
            ...
        }
    }
}
```

## Comparison with Centralized Results

### Centralized (NCF):
```json
{
    "model_name": "NCF",
    "rmse": 0.8758904476594858,
    "mae": 0.6904203345112558,
    "mse": 0.7671840763011343,
    "r2": 0.38810927822753494,
    "training_time": 72.050985
}
```

### Federated (MF):
```json
{
    "model_name": "BPR_MF_Federated",
    "dataset": "ml-1m",
    "final_metrics": {
        "rmse": 0.9157,
        "mae": 0.7319,
        "hit_rate@10": 0.4329,
        "precision@10": 0.0645,
        "ndcg@10": 0.0683,
        ...
    },
    "federated_config": {...},
    "metrics_per_round": {...}
}
```

## Key Differences

1. **Centralized**: Single training, single result
2. **Federated**:
   - Multiple rounds
   - Distributed training
   - Per-round metrics tracking
   - Additional ranking metrics (hit rate, precision, recall, NDCG, MRR)

## Benefits of This Format

✅ **Consistent structure** with centralized results
✅ **Easy comparison** between centralized and federated approaches
✅ **Rich metrics** including ranking evaluation
✅ **Traceable** with per-round performance
✅ **Reproducible** with full config saved
