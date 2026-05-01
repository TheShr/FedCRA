# FedCRA Client Tuning for 5 vs 10 Clients

## Summary

This note documents the current FedCRA hyperparameter tuning for `num_clients=5` and `num_clients=10`, and records the code fix applied to ensure client-side FedCRA training receives `lambda_cra` correctly.

## Code fix

- Fixed `src/fedLearn/strategies/fedcra_strategy.py` so the server now sends `lambda_cra` to clients in `configure_fit()`.
- Without this fix, per-client FedCRA training always used the default CRA coefficient (`0.1`), even when the tuning script overrode `strategy.params.lambda_cra`.

## Recommended FedCRA tuning

### Client = 5

For stable FedCRA performance with 5 clients and highly skewed data:

- `strategy.params.proximal_mu = 0.015`
- `strategy.params.lambda_cra = 0.12`
- `strategy.params.embedding_dim = 128`
- `config_fit.grad_clip = 1.5`
- `cra_grad_clip = 1.5`
- `fed_config.num_clients_per_round_fit = 5`
- `fed_config.num_clients_per_round_eval = 5`
- `fed_config.num_client_local_rounds = 1`
- `fed_config.learning_rate = 0.001`

### Client = 10

For 10 clients, FedCRA must be tuned more conservatively than the 5-client case because the CRA anchor penalty can overwhelm rare classes when heterogeneity increases.

- `strategy.params.proximal_mu = 0.01`
- `strategy.params.lambda_cra = 0.08`
- `strategy.params.embedding_dim = 128`
- `config_fit.grad_clip = 1.5`
- `cra_grad_clip = 1.5`
- `fed_config.num_clients_per_round_fit = 10`
- `fed_config.num_clients_per_round_eval = 10`
- `fed_config.num_client_local_rounds = 1`
- `fed_config.learning_rate = 0.001`

> Note: FedCRA is an extension of FedProx, but its additional anchor regularization must be milder for 10 clients. The server now passes the actual `lambda_cra` to clients, and the 10-client tuning should be lower than the 5-client setting to avoid harming minority classes.

## Full participation policy

The current experiment launcher now uses full participation for every configuration:
- `num_clients=5` → `num_clients_per_round_fit=5`, `num_clients_per_round_eval=5`
- `num_clients=10` → `num_clients_per_round_fit=10`, `num_clients_per_round_eval=10`

This matches the expectation that all clients participate in each round.

## run_heterogeneity_experiments.sh check

- `run_heterogeneity_experiments.sh` already applies per-alpha FedCRA tuning and 50% participation for 10 clients.
- The new fix makes those hyperparameter overrides effective for client-side FedCRA training.
- The script should now work nicely for both 5- and 10-client experiments, provided the `lambda_cra` and `proximal_mu` settings are set correctly.

## How to validate

1. Run the 10-client FedCRA experiment with `alpha=0.1`.
2. Confirm `server_metrics.json` shows non-zero per-class F1 values for all classes.
3. Compare against the 5-client configuration to ensure the same FedCRA coefficients are being used.
