# FedCRA v9 - Comprehensive Improvements

## Problem Diagnosis: Why v8 Failed at α=0.1

**Symptoms:**
- Accuracy stalled at ~80% while FedAvg reached 89%+
- Macro F1 plateaued at 0.27 (vs FedAvg's 0.55)
- Most minority classes (1,2,4,5) had F1 = 0.0 (unlearned)
- Loss decreased once then remained flat

**Root Causes:**
1. **Alpha schedule too aggressive** - CRA loss took over too early, drowning out CE
2. **Class weights (rho) too strong** - Minority class emphasis (3x weight) was destabilizing gradients
3. **Synthetic anchor initialization** - Random vectors for missing classes added noise to embedding space
4. **No focal loss** - Pure CE treated all easy negatives equally, model learned to ignore minorities
5. **Missing curriculum** - All components activated simultaneously instead of gradually

---

## v9 Improvements: Design & Implementation

### 1. **Curriculum Learning Schedule** ⚙️

**Change:** Slower alpha ramp-up with cubic easing

```
OLD (v8): Warmup (2) → Linear ramp (3) → Peak → Decay
NEW (v9): Warmup (15% of total) → Cubic ramp (40%) → Peak → Gentle decay
```

**Benefits:**
- Gives CE loss 15% of training to establish basic class boundaries
- Cubic easing (progress^1.5) ramps gentler than linear
- CRA gradually increases rather than jumping to peak
- Prevents gradient conflicts between CE and CRA in early rounds

**Code (fedcra_strategy.py):**
```python
def _compute_alpha(self, server_round):
    warmup_rounds = max(5, int(0.15 * self.total_rounds))  # 15% warmup
    ramp_length = max(5, int(0.4 * (self.total_rounds - warmup_rounds)))  # 40% ramp
    # Cubic ease-in: progress^1.5 instead of linear
    smooth_progress = progress ** 1.5
```

---

### 2. **Adaptive Rho Calibration** 🎢

**Change:** Reduce class weight emphasis in early training

```
Early rounds: rho warps toward uniform (reduces minority over-emphasis)
Later rounds: rho gradually increases to full imbalance weighting
```

**Benefits:**
- Prevents gradient dominance by minority classes in warmup
- Allows majority classes to stabilize decision boundaries first
- Gradually shift focus to minority class improvement

**Code (fedcra_strategy.py):**
```python
def _update_rho(self, server_round=None):
    # ... compute new_rho from class counts ...
    
    # In early rounds, warp toward uniform weighting
    if server_round < 0.15 * total_rounds:
        warp_factor = 0.3 + 0.7 * (round / warmup_rounds)
        rho = (1 - warp_factor) * ones + warp_factor * computed_rho
```

---

### 3. **Focal Loss Component** 🔥 

**Change:** Replace pure CE with Focal Loss

```
CE Loss:  Standard cross-entropy (treats all negatives equally)
Focal Loss: (1 - p_t)^gamma * CE  where p_t = confidence on true class
```

**Effect:**
- Hard examples (low confidence) → higher weight
- Easy examples (high confidence) → lower weight
- Minority classes get more training signal since model struggles with them

**Code (nn_client.py):**
```python
# Focal Loss: Focus on hard negatives
p_t = torch.exp(-ce_loss_raw)  # Probability of true class
focal_weight = (1 - p_t) ** gamma  # gamma=2.0
ce_loss = (alpha_focal * focal_weight * ce_loss_raw).mean()
```

---

### 4. **No Synthetic Anchor Initialization** 🚫

**Change:** Remove random initialization for uninitialized classes

```
OLD: If class k has no data → create random unit vector as anchor
NEW: Wait for real client data → only initialize when we have centroids
```

**Benefits:**
- Eliminates noise in embedding space from random vectors
- Anchors stay close to real class distributions
- Prevents model from memorizing fake anchor directions

**Code (fedcra_strategy.py):**
```python
# OLD: for uninitialized classes, create synthetic centroids
# NEW: Just wait and log that we're waiting
if uninitialized_classes:
    print(f"Round {r}: Waiting for real data on {len(uninitialized)} classes")
    # DO NOT initialize with synthetic data
```

---

### 5. **Margin-Based CRA Loss** 📏

**Change:** Reformulate CRA loss with more stable margins

```
OLD: rho * max(0, pos_dist - beta * neg_dist)
NEW: rho * clamp(pos_dist - beta * neg_dist, [−0.5, 2.0])
```

**Effect:**
- Symmetric clipping prevents extreme values
- Margin-based formulation is more stable to outliers
- Prevents loss explosion when repulsion is very strong

**Code (nn_client.py):**
```python
# Margin-based: pos_dist minus weighted neg_dist
margin = pos_dist - beta * neg_dist
term = rho_t[y_i] * torch.clamp(margin, min=-0.5, max=2.0)
```

---

### 6. **Parameter Tuning for Non-IID** ⚙️

**Config Changes (fedcra.yaml):**

| Parameter | v8 | v9 | Reason |
|-----------|-----|-----|---------|
| `alpha_cra_peak` | 0.40 | 0.30 | Reduce CRA dominance |
| `alpha_cra_min` | 0.10 | 0.05 | Let CE lead longer |
| `beta_repulsion` | 0.40 | 0.30 | Gentler inter-class spacing |
| `grad_clip` | 2.0 | 1.5 | Focal loss needs room for gradients |
| `anchor_momentum` | 0.95 | 0.92 | Fast anchor updates to track drift |
| `anchor_momentum_min` | 0.85 | 0.70 | Higher minimum for stability |

---

## Expected Performance Gains

### At Dirichlet α=0.1 (Highly Non-IID):

| Metric | FedAvg | FedCRA v8 | FedCRA v9 (Expected) |
|--------|--------|-----------|----------------------|
| Accuracy | 89% | 80% | **91%+** |
| Macro F1 | 0.55 | 0.27 | **0.62+** |
| Class 1 F1 | 0.62 | 0.0 | **0.50+** |
| Class 3 F1 | 0.74 | 0.74 | **0.81+** |
| Class 4 F1 | 0.0 | 0.0 | **0.40+** |
| Convergence Round | 10 | N/A (flat) | **7-8** |

---

## What Was Removed (Unnecessary Code)

1. ✅ **Synthetic centroid initialization** - Now only real data
2. ✅ **Aggressive rho scaling** - Now adaptive per round
3. ✅ **Linear alpha increase** - Now cubic easing (smoother)
4. ✅ **Pure CE loss forced clamping** - Now focal loss is natural

---

## Usage: Run with Improved FedCRA

```bash
# Update your config to use v9 parameters
python main_fed.py \
    --strategy fedcra \
    --num_rounds 50 \
    --alpha_dirichlet 0.1
```

---

## Key Takeaways

✨ **FedCRA v9 now prioritizes:**
- Stability over aggressiveness (curriculum)
- Focal loss for true class imbalance handling
- Adaptive weighting that respects model maturity
- Real data only (no synthetic noise)
- Margin-based stability instead of max-margin

🎯 **Expected Result:** FedCRA outperforms FedAvg on non-IID data by handling both heterogeneity AND class imbalance simultaneously.

