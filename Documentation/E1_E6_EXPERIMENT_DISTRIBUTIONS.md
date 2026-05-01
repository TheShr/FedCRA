# CIC-IoMT Experiment Distributions and Oversampling

## Current dataset and class set

The current CIC-IoMT dataset in this repository is configured to use 6 traffic classes:

- `Normal`
- `DoS`
- `DDoS`
- `Mirai`
- `ARP`
- `Recon`

The repository uses `data_config.total_samples` as the target dataset size for experiment creation.

## What happens if 100k samples are selected?

When `data_config.total_samples = 100000` and `imbalance.class_ratios` is provided, the pipeline does the following:

1. Load the raw CIC-IoMT dataset.
2. Keep only the first `n_features` columns plus the label column.
3. Compute target class counts from the configured `class_ratios`.
4. Use `create_controlled_dataset(..., total_samples=100000, method='oversample')`.

The result is a new dataset of exactly `100000` examples.

## How oversampling works

The current implementation uses class-level sampling with replacement when necessary.

- If the requested target count for a class is less than or equal to the number of raw examples for that class, the generator samples without replacement.
- If the requested target count is greater than the number of available examples for that class, the generator samples with replacement.

That means the dataset can still be built to exactly `100000` examples even if one or more classes are rare in the raw source.

### Example

If `DDoS` is requested to have 15,000 samples but only 7,500 raw `DDoS` rows exist, the code will sample 7,500 unique rows and then continue sampling additional `DDoS` rows with replacement until the class reaches 15,000 samples.

This preserves the requested class presence and ratio while ensuring the final dataset size is exact.

## Experiment definitions E1..E6

The heterogeneity experiment matrix is defined in `run_heterogeneity_experiments.sh` as:

| Experiment | Imbalance profile | Dirichlet alpha |
|-----------:|------------------:|----------------:|
| E1 | balanced | 1.0 |
| E2 | mild | 1.0 |
| E3 | severe | 1.0 |
| E4 | balanced | 0.1 |
| E5 | severe | 0.1 |
| E6 | extreme | 0.1 |

## Class ratio distributions for 100k samples

### E1 — balanced / alpha=1.0

`conf/imbalance/balanced.yaml`

```yaml
class_ratios:
  Normal: 1
  DoS: 1
  DDoS: 1
  Mirai: 1
  ARP: 1
  Recon: 1
```

With 100k total samples, this targets roughly equal class sizes:

- `Normal`: 16,667
- `DoS`: 16,667
- `DDoS`: 16,667
- `Mirai`: 16,667
- `ARP`: 16,667
- `Recon`: 16,665

### E2 — mild / alpha=1.0

`conf/imbalance/mild.yaml`

```yaml
class_ratios:
  Normal: 10
  DoS: 8
  DDoS: 6
  Mirai: 4
  ARP: 2
  Recon: 1
```

Total ratio weight = 31. Example target counts:

- `Normal`: ~32,258
- `DoS`: ~25,806
- `DDoS`: ~19,355
- `Mirai`: ~12,903
- `ARP`: ~6,452
- `Recon`: ~3,226

### E3 — severe / alpha=1.0

`conf/imbalance/severe.yaml`

```yaml
class_ratios:
  Normal: 50
  DoS: 25
  DDoS: 15
  Mirai: 7
  ARP: 2
  Recon: 1
```

Total ratio weight = 100. Example target counts:

- `Normal`: 50,000
- `DoS`: 25,000
- `DDoS`: 15,000
- `Mirai`: 7,000
- `ARP`: 2,000
- `Recon`: 1,000

### E4 — balanced / alpha=0.1

Same class ratio as E1, but with stronger client heterogeneity.

- At `alpha=0.1`, the Dirichlet split produces much more skewed client data distributions.
- Global class ratios remain the same as E1, but clients see fewer classes and more concentrated per-client distributions.

### E5 — severe / alpha=0.1

Same class ratio as E3, with extreme client-level skew.

- The global dataset is still highly imbalanced in the same proportions as E3.
- The low alpha makes each client more likely to receive only a few dominant class labels.

### E6 — extreme / alpha=0.1

`conf/imbalance/extreme.yaml`

```yaml
class_ratios:
  Normal: 70
  DoS: 20
  DDoS: 10
  Mirai: 4
  ARP: 1
  Recon: 1
```

Total ratio weight = 106. Example target counts:

- `Normal`: ~66,038
- `DoS`: ~18,868
- `DDoS`: ~9,434
- `Mirai`: ~3,774
- `ARP`: ~943
- `Recon`: ~943

This is the most extreme global imbalance in the matrix.

## Alpha explanation

- `alpha=1.0`: moderate Dirichlet non-IID, each client gets a mix of classes.
- `alpha=0.1`: strong non-IID, clients tend to receive highly skewed class subsets.

So E1/E2/E3 compare global imbalance strength under moderate heterogeneity, while E4/E5/E6 compare the same global imbalance under much stronger client heterogeneity.

## What this means for the current CIC-IoMT dataset

If the raw CIC-IoMT data contains all 6 classes, the controlled dataset creation will slice or oversample each class to match the target counts above.
If any class is too rare for a requested target count, the code oversamples that class with replacement until the target is met.

If the raw dataset does not contain a requested class label at all, the current implementation logs the missing class and matches only the labels present in the dataset.

## Practical effect

- E1/E4: nearly equal global class balance.
- E2: mild global class imbalance.
- E3: strong global class imbalance.
- E6: extreme global class imbalance.
- E4/E5/E6: stronger per-client heterogeneity than E1/E2/E3.

This document explains the expected distribution of the six heterogeneity experiments for a 100k sample CIC-IoMT run.
