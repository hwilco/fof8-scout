# Draft Value Target Strategy

This document defines how to choose regression targets and evaluation metrics for the
FOF8 draft model. The goal is not only to predict player quality, but to build a draft
board that helps decide who to pick, when to trade, and how much expected value is left
on the board.

## Product Goal

The model should support draft decisions in a game that mimics NFL roster building:

- Rank prospects within a live draft class.
- Estimate the magnitude of expected value for trade decisions.
- Surface upside while avoiding expensive busts.
- Compare players across positions without hiding positional context.

This means target and metric choices should be judged by draft-board usefulness, not by
plain global regression error.

## Current DPO Baseline

The current composite baseline target is:

```text
DPO = Peak_Overall * Career_Merit_Cap_Share
```

`Peak_Overall` is a talent/scouted rating proxy. `Career_Merit_Cap_Share` is an
economic outcome that subtracts expected rookie-contract cost from actual career cap share.

### What DPO Gets Right

- It rewards players who became highly rated and economically meaningful.
- It uses the game economy to encode some positional value.
- It is nonzero mostly for players who matter, which fits the hurdle model.

### Problems With DPO

- It mixes two different concepts: ability rating and market compensation.
- It is nearly a scaled version of `Career_Merit_Cap_Share`, so the peak-rating multiplier
  may add conceptual complexity without much independent value.
- `Peak_Overall` is a scouted/rating outcome, not direct production.
- `Career_Merit_Cap_Share` can reward market behavior and longevity rather than cheap
  draft surplus.
- It is hard to explain as a trade-value unit.

Recommendation: keep DPO as a composite baseline/comparison target, but do not treat it as the
long-term primary regression target without target-selection experiments.

## Overall-Based Vs Economic Targets

Overall-based targets and economic targets answer different questions.

### Overall-Based Targets

Examples:

- `Peak_Overall`
- mean of top 3 `Current_Overall` observations
- years above a position-specific starter overall threshold

Use these when the question is:

```text
Who became the best football player according to the game/scouting model?
```

Pros:

- Easier to explain as talent.
- Less directly affected by contract quirks.
- Comparable on a familiar 0-100 style scale.

Cons:

- Still based on scouted ratings, not pure production.
- May understate positional economics.
- Does not directly answer trade value or draft surplus.

### Economic Targets

Examples:

- clipped positive `Career_Merit_Cap_Share`
- post-rookie cap share
- second-contract cap share
- rookie-contract surplus value

Use these when the question is:

```text
Who produced value that the game economy recognized?
```

Pros:

- Naturally cross-position through the game salary market.
- Easier to connect to trade value and draft capital.
- Aligns with roster-building decisions.

Cons:

- Can reflect market quirks rather than on-field value.
- Can reward compilers.
- May overemphasize premium positions if not inspected carefully.

## Why Not Hand-Built AV As The Primary Target

The repo has Approximate Value and VORP machinery, but the current AV formula is
hand-weighted. That makes it hard to defend as a universal target across positions.

A QB's passing production, an OL's start/block proxy, and a CB's turnover/tackle line do
not naturally share a scale unless the weights are calibrated. Until that calibration is
done, AV/VORP is useful as research, but it should not automatically replace economic or
overall targets as the primary production target.

If AV is used later, prefer a calibration step:

- Fit or tune AV weights against game-recognized outcomes such as salary, starts, awards,
  HOF points, or overall.
- Validate by position, not only globally.
- Confirm that top-ranked AV players pass manual football sanity checks.

## Recommended Target Experiments

Run target selection as an empirical experiment. Keep features, train/test split, model
family, and sweep budget fixed while changing only the target.

Candidate targets:

```text
1. Peak_Overall
2. Top3_Mean_Current_Overall
3. clipped_positive_Career_Merit_Cap_Share
4. post_rookie_or_second_contract_cap_share
5. DPO baseline
```

For each target, train the same classifier/regressor architecture and evaluate the
resulting draft board on multiple held-out definitions of success:

- actual target value
- `Peak_Overall`
- `Career_Merit_Cap_Share`
- `Career_Games_Played`
- awards/HOF labels if available

A strong target should not only predict itself. It should create draft boards that look
good under several reasonable definitions of "good player."

## Recommended Modeling Direction

The preferred near-term setup is an economic primary score plus talent diagnostics:

```text
Classifier:
  P(player becomes economically meaningful)
  e.g. Career_Merit_Cap_Share > 0

Regressor:
  E(positive economic value | economically meaningful, X)
  e.g. clipped_positive_Career_Merit_Cap_Share

Complete score:
  P(success | X) * E(value | success, X)
```

Display secondary signals next to the primary score:

- predicted `Peak_Overall`
- position-relative percentile
- uncertainty/risk bucket
- classifier probability

This keeps the board's primary sort tied to draft economics while still exposing talent
information when economic and overall-based predictions disagree.

## Evaluation Metrics For Regressors

Do not use global RMSE as the primary sweep metric. It is useful as a diagnostic, but it
does not match how the model is used.

Draft decisions need both:

- ordering quality: who should be above whom?
- magnitude quality: how much value is the player worth for trade decisions?

### Ranking Metrics

Use draft-class grouped ranking metrics:

```text
mean_ndcg_at_32
mean_ndcg_at_64
mean_ndcg_at_128
```

NDCG should be computed within each draft class and then averaged across classes. Use
clipped non-negative target value as relevance.

### Magnitude Metrics

Use magnitude metrics focused on the decision region instead of the entire draft pool:

```text
top64_weighted_mae
top64_weighted_mae_normalized
top64_bias
top64_calibration_slope
```

The model should be penalized for being badly calibrated on players it ranks highly,
because those are the players that drive picks and trades.

### Recommended Primary Sweep Metric

If the sweep framework needs one scalar, use a composite:

```text
score = mean_ndcg_at_64 - 0.25 * top64_weighted_mae_normalized
```

Maximize this score.

This keeps ranking primary while penalizing bad magnitude estimates. The `0.25` weight is
a starting point, not a law. Tune it after inspecting whether the model is over-optimizing
ranking at the expense of useful value estimates.

If a minimize-only metric is easier:

```text
loss = top64_weighted_mae_normalized - 4.0 * mean_ndcg_at_64
```

Minimize this loss. It is algebraically equivalent up to scale.

### Metrics To Log Alongside The Sweep Objective

Always log:

```text
regressor_rmse_positive
regressor_mae_positive
regressor_spearman_by_draft_class
regressor_mean_ndcg_at_32
regressor_mean_ndcg_at_64
regressor_top64_weighted_mae
regressor_top64_weighted_mae_normalized
regressor_top64_bias
regressor_top64_calibration_slope
```

For stitched classifier + regressor model selection, also log:

```text
complete_mean_ndcg_at_32
complete_mean_ndcg_at_64
complete_top32_actual_value
complete_top64_actual_value
complete_top32_weighted_mae_normalized
complete_bust_rate_at_32
complete_precision_at_32_positive_value
```

The standalone regressor sweep can optimize conditional quality on the positive/success
subset. Final champion selection should use the complete stitched model on the full draft
class.

## Training Loss Functions

Separate the model's training loss from the model-selection metric.

CatBoost needs a differentiable loss during fitting, but champion selection should still
use draft-aware held-out metrics such as NDCG, top-k weighted MAE, bias, and calibration.
The loss should be treated as a candidate modeling choice, not as the definition of
success.

### Losses By Target Type

For clipped non-negative economic value:

```text
target = max(Career_Merit_Cap_Share, 0)
loss_function = Tweedie
```

Tweedie is a strong first candidate for economic value because the target is non-negative,
right-skewed, and often zero-heavy. CatBoost requires Tweedie labels to be non-negative,
which is why the target should be clipped or otherwise constrained before training.

For positive-only conditional economic value:

```text
target = Career_Merit_Cap_Share where Career_Merit_Cap_Share > 0
loss candidates = Tweedie, MAE, RMSE on log1p(target), Expectile
```

If the regressor is trained only on players that clear the classifier threshold, the
target is less zero-inflated. Tweedie can still work, but its advantage over log-RMSE or
MAE is less guaranteed.

For overall-based talent targets:

```text
target = Peak_Overall or Top3_Mean_Current_Overall
loss candidates = RMSE, MAE
```

Overall targets are bounded, continuous-ish rating outcomes. RMSE is a reasonable
baseline; MAE is more robust if scouting artifacts or outliers are an issue.

For upside-aware models:

```text
loss candidates = Quantile:alpha=0.7, Expectile:alpha=0.7
```

These losses intentionally bias the model toward the upper tail. They can be useful when
missing stars is more costly than slightly overrating ordinary players, but they may hurt
trade-value calibration.

### Recommended Loss Sweep

For economic targets, compare at least these two families:

```text
1. Raw clipped target + Tweedie
2. log1p clipped target + RMSE
```

Optional additional candidates:

```text
3. Raw clipped target + MAE
4. Raw clipped target + Expectile:alpha=0.6-0.8
5. Raw clipped target + Quantile:alpha=0.6-0.8
```

Expected behavior:

- Tweedie: strong for non-negative, heavy-tailed economic outcomes.
- log1p + RMSE: often stable and well-calibrated after `expm1`.
- MAE: robust, but can underfit rare stars.
- Expectile/Quantile: more upside-sensitive, but can overstate value.

### CatBoost Config Examples

Raw economic value with Tweedie:

```yaml
name: "catboost_tweedie_regressor"
params:
  loss_function: "Tweedie"
  variance_power: 1.25
  eval_metric: "Tweedie"
```

Log-transformed economic value with RMSE:

```yaml
name: "catboost_log_rmse_regressor"
target_space: "log"
params:
  loss_function: "RMSE"
  eval_metric: "RMSE"
```

Upside-aware conditional value:

```yaml
name: "catboost_expectile_regressor"
params:
  loss_function: "Expectile:alpha=0.7"
  eval_metric: "MAE"
```

### Selection Rule

Do not pick a loss because its built-in validation loss is lowest. Pick the loss family
whose held-out draft-board metrics are best:

```text
primary:
  mean_ndcg_at_64 - 0.25 * top64_weighted_mae_normalized

guardrails:
  top64_bias near 0
  reasonable calibration slope
  no severe position-specific failure modes
```

For the current economic direction, the first loss to try is Tweedie on raw clipped value.
The main challenger should be RMSE on `log1p` value. Let the complete stitched model's
held-out draft metrics decide.

## Metric Definitions

### NDCG@K By Draft Class

```python
import numpy as np


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    gains = np.maximum(y_true[order], 0)

    discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains * discounts)

    ideal = np.sort(np.maximum(y_true, 0))[::-1][:k]
    ideal_dcg = np.sum(ideal * discounts)

    return 0.0 if ideal_dcg == 0 else float(dcg / ideal_dcg)


def mean_ndcg_by_draft_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    draft_year: np.ndarray,
    k: int,
) -> float:
    scores = []
    for year in np.unique(draft_year):
        mask = draft_year == year
        scores.append(ndcg_at_k(y_true[mask], y_score[mask], k=k))
    return float(np.mean(scores))
```

### Top-K Weighted MAE

```python
def topk_weighted_mae(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    order = np.argsort(y_score)[::-1][:k]
    errors = np.abs(y_true[order] - y_score[order])
    ranks = np.arange(1, len(order) + 1)
    weights = 1 / np.log2(ranks + 1)
    return float(np.sum(errors * weights) / np.sum(weights))
```

Normalize it by the scale of the actual top-K opportunity:

```python
def topk_weighted_mae_normalized(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    eps: float = 1e-9,
) -> float:
    mae = topk_weighted_mae(y_true, y_score, k=k)
    order = np.argsort(y_score)[::-1][:k]
    scale = float(np.mean(np.maximum(y_true[order], 0)))
    return mae / max(scale, eps)
```

### Top-K Bias

```python
def topk_bias(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_score[order] - y_true[order]))
```

Positive bias means the model overstates value in the players it recommends. Negative
bias means it understates value and may make good trade-down opportunities look less
valuable than they are.

### Calibration Slope

Fit a simple held-out calibration line:

```text
y_true ~= intercept + slope * y_pred
```

Interpretation:

- `slope < 1`: predictions are too spread out; high predictions are likely too high.
- `slope > 1`: predictions are too compressed; high predictions may be too conservative.
- nonzero intercept: systematic offset.

Use this as a diagnostic, not the sole optimization target.

## Decision Rule

Choose the primary target and sweep metric based on which model produces the best draft
board, not the best global fit.

Recommended initial decision:

```text
Primary target:
  clipped_positive_Career_Merit_Cap_Share

Primary sweep objective:
  mean_ndcg_at_64 - 0.25 * top64_weighted_mae_normalized

Final champion selection:
  complete stitched classifier + regressor metrics on full held-out draft classes

Secondary model/output:
  predicted Peak_Overall or position-relative talent percentile
```

Revisit this after running target-selection experiments. If an overall-based target
consistently produces better economic draft outcomes and better manual board reviews, use
it. If the economic target finds lower-overall but higher-value players, keep economic
value as the primary sort and display overall as context.
