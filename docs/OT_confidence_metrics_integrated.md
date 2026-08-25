# OT Confidence Metrics and Recommended Settings

## Overview

Optimal transport (OT) does not inherently provide a calibrated confidence score.  
Instead, confidence can be constructed from the OT coupling matrix, transport cost, or the stability of the matching result.

Let

\[
P \in \mathbb{R}^{N_{\mathrm{src}}\times N_{\mathrm{tgt}}}
\]

be the OT coupling matrix. For each source cell \(i\), normalize its row as

\[
p_{ij}
=
\frac{P_{ij}}
{\sum_j P_{ij}+\epsilon}.
\]

Unless otherwise stated, the recommended numerical constant is

\[
\epsilon = 10^{-8}.
\]

---

## 1. Maximum-Coupling Confidence

The simplest confidence definition is

\[
C_i^{\max}
=
\max_j p_{ij}.
\]

### Interpretation

- High value: most transport mass is assigned to one target.
- Low value: transport mass is distributed across many targets.

### Recommended use

Use this as the baseline because it is simple and directly comparable with the SPATIA formulation.

### Suggested interpretation thresholds

| Score | Interpretation |
|---|---|
| \(C_i^{\max} \ge 0.80\) | Very concentrated |
| \(0.60 \le C_i^{\max} < 0.80\) | Moderately concentrated |
| \(0.40 \le C_i^{\max} < 0.60\) | Ambiguous |
| \(C_i^{\max} < 0.40\) | Highly diffuse |

These are heuristic thresholds and should be recalibrated for each dataset.

---

## 2. Entropy-Based Confidence

Compute the row entropy:

\[
H_i
=
-\sum_j p_{ij}\log(p_{ij}+\epsilon).
\]

Then normalize it:

\[
C_i^{\mathrm{entropy}}
=
1-
\frac{H_i}
{\log N_{\mathrm{tgt}}}.
\]

### Interpretation

- \(C_i^{\mathrm{entropy}}\approx 1\): coupling is highly concentrated.
- \(C_i^{\mathrm{entropy}}\approx 0\): coupling is close to uniform.

### Recommended range

\[
C_i^{\mathrm{entropy}} \in [0,1].
\]

### Suggested interpretation thresholds

| Score | Interpretation |
|---|---|
| \(\ge 0.75\) | High confidence |
| \(0.50\)–\(0.75\) | Moderate confidence |
| \(0.25\)–\(0.50\) | Low confidence |
| \(<0.25\) | Very diffuse matching |

### Recommendation

Use entropy-based confidence as the main concentration metric because it uses the entire OT row rather than only the largest entry.

---

## 3. Top-1 versus Top-2 Margin

Let \(p_{i(1)}\) and \(p_{i(2)}\) be the largest and second-largest values in row \(i\):

\[
C_i^{\mathrm{margin}}
=
p_{i(1)}-p_{i(2)}.
\]

### Interpretation

- Large margin: one target is clearly preferred.
- Small margin: the top two targets are similarly plausible.

For example,

\[
[0.50,0.49,0.01]
\]

has a maximum of \(0.50\), but a margin of only \(0.01\), indicating strong ambiguity.

### Suggested thresholds

| Margin | Interpretation |
|---|---|
| \(\ge 0.30\) | Clear dominant match |
| \(0.15\)–\(0.30\) | Moderate separation |
| \(0.05\)–\(0.15\) | Weak separation |
| \(<0.05\) | Highly ambiguous |

### Recommendation

Use this together with maximum coupling. It detects cases in which the maximum coupling is not sufficiently dominant.

---

## 4. Top-1 versus Top-2 Ratio

Define

\[
C_i^{\mathrm{ratio}}
=
\frac{p_{i(1)}}
{p_{i(2)}+\epsilon}.
\]

### Interpretation

- Large ratio: the best target is clearly preferred over the second-best.
- Ratio near \(1\): the top two targets are almost equally likely.

### Suggested thresholds

| Ratio | Interpretation |
|---|---|
| \(\ge 3.0\) | Strongly dominant target |
| \(2.0\)–\(3.0\) | Moderate dominance |
| \(1.2\)–\(2.0\) | Weak dominance |
| \(<1.2\) | Nearly tied |

### Recommendation

Use the log-transformed ratio for modeling:

\[
\widetilde{C}_i^{\mathrm{ratio}}
=
\log\left(C_i^{\mathrm{ratio}}\right).
\]

This reduces sensitivity to very large ratios.

---

## 5. Effective Number of Targets

Using the entropy, define

\[
N_{\mathrm{eff},i}
=
\exp(H_i).
\]

This estimates how many target cells effectively receive transport mass.

A corresponding confidence score is

\[
C_i^{\mathrm{eff}}
=
\frac{1}{N_{\mathrm{eff},i}}.
\]

A normalized version is

\[
\widetilde{C}_i^{\mathrm{eff}}
=
1-
\frac{N_{\mathrm{eff},i}-1}
{N_{\mathrm{tgt}}-1}.
\]

### Interpretation

- \(N_{\mathrm{eff},i}\approx 1\): nearly one-to-one mapping.
- Large \(N_{\mathrm{eff},i}\): diffuse mapping.

### Recommendation

Report \(N_{\mathrm{eff}}\) as an interpretable statistic and use the normalized version in a combined confidence model.

---

## 6. Transport-Cost Confidence

Let \(c_{ij}\) be the OT cost between source cell \(i\) and target cell \(j\). Compute the expected transport cost:

\[
\bar{c}_i
=
\sum_j p_{ij}c_{ij}.
\]

Convert it to confidence:

\[
C_i^{\mathrm{cost}}
=
\exp\left(
-\frac{\bar{c}_i}{\tau}
\right).
\]

### Choosing \(\tau\)

A robust default is

\[
\tau
=
\operatorname{median}_i(\bar{c}_i).
\]

Alternative choices:

\[
\tau
\in
\left\{
0.5\,\operatorname{median}(\bar{c}),
\operatorname{median}(\bar{c}),
2\,\operatorname{median}(\bar{c})
\right\}.
\]

### Recommendation

Start with

\[
\tau=\operatorname{median}(\bar{c}).
\]

This is more transferable across datasets than choosing a fixed absolute value.

---

## 7. Bootstrap or Stability Confidence

Repeat OT under small perturbations, such as:

- bootstrap resampling of cells;
- different random seeds;
- different gene subsets;
- different PCA dimensions;
- different entropy regularization strengths;
- slightly different OT hyperparameters.

Let \(R\) be the number of repeated OT runs. A simple stability score is

\[
C_i^{\mathrm{stability}}
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbb{I}
\left[
j_i^{(r)}
\in
\mathcal{N}_k(j_i^{\mathrm{ref}})
\right],
\]

where \(j_i^{(r)}\) is the selected target in run \(r\), and \(\mathcal{N}_k\) is the \(k\)-nearest-target neighborhood around the reference match.

### Recommended values

- Number of repeats:

\[
R = 20
\]

for an initial experiment.

- More robust evaluation:

\[
R = 50
\]

if computationally affordable.

- Neighborhood size:

\[
k = 5
\]

for cell-level matching, or

\[
k = 10
\]

for noisy datasets.

### Suggested interpretation

| Stability | Interpretation |
|---|---|
| \(\ge 0.80\) | Highly stable |
| \(0.60\)–\(0.80\) | Moderately stable |
| \(0.40\)–\(0.60\) | Uncertain |
| \(<0.40\) | Unstable |

### Recommendation

Use stability confidence as the strongest uncertainty signal when computational cost allows.

---

## 8. Mutual or Cycle-Consistency Confidence

For each source cell \(i\):

1. Find its target match \(j\).
2. Perform reverse target-to-source matching.
3. Test whether the reverse match returns to \(i\) or its local neighborhood.

A binary version is

\[
C_i^{\mathrm{cycle}}
=
\mathbb{I}
\left[
i
=
\operatorname{argmax}_{i'}
P^{\mathrm{reverse}}_{ji'}
\right].
\]

A neighborhood-based version is preferable:

\[
C_i^{\mathrm{cycle}}
=
\mathbb{I}
\left[
i
\in
\mathcal{N}_k(i_j^{\mathrm{reverse}})
\right].
\]

### Recommended value

Use

\[
k=5
\]

as the default neighborhood size.

### Recommendation

Use cycle consistency when approximate one-to-one biological correspondence is expected. It may be less suitable for genuinely many-to-many transitions.

---

## 9. Unbalanced-OT Retained-Mass Confidence

For unbalanced OT, let \(a_i\) be the original mass of source cell \(i\). Define

\[
C_i^{\mathrm{mass}}
=
\frac{\sum_j P_{ij}}
{a_i+\epsilon}.
\]

### Interpretation

- Value near \(1\): most source mass found a target.
- Low value: source cell may not have a convincing target counterpart.

### Suggested thresholds

| Retained mass | Interpretation |
|---|---|
| \(\ge 0.90\) | Strongly matched |
| \(0.70\)–\(0.90\) | Moderately matched |
| \(0.50\)–\(0.70\) | Weakly matched |
| \(<0.50\) | Potentially unmatched |

### Recommendation

Use this metric only when fitting unbalanced OT. It is not informative under strictly balanced OT because row mass is constrained.

---

# Recommended Metric Set

For an initial study, use the following five metrics:

\[
\boxed{
C_i^{\max},
\quad
C_i^{\mathrm{entropy}},
\quad
C_i^{\mathrm{margin}},
\quad
C_i^{\mathrm{cost}},
\quad
C_i^{\mathrm{stability}}
}
\]

This set covers:

- peak concentration;
- whole-distribution concentration;
- separation between the top candidates;
- biological or geometric proximity;
- robustness to perturbations.

---

# Recommended Combined Confidence

Before combining metrics, normalize each metric to \([0,1]\), preferably using percentile or rank normalization.

A simple weighted average is

\[
C_i^{\mathrm{combined}}
=
\alpha_1 C_i^{\mathrm{entropy}}
+
\alpha_2 C_i^{\mathrm{cost}}
+
\alpha_3 C_i^{\mathrm{stability}}.
\]

Recommended initial weights:

\[
\boxed{
\alpha_1=0.35,
\qquad
\alpha_2=0.25,
\qquad
\alpha_3=0.40
}
\]

These weights prioritize stability, followed by coupling concentration and matching cost.

An alternative multiplicative form is

\[
C_i^{\mathrm{combined}}
=
\left(C_i^{\mathrm{entropy}}\right)^{0.35}
\left(C_i^{\mathrm{cost}}\right)^{0.25}
\left(C_i^{\mathrm{stability}}\right)^{0.40}.
\]

The multiplicative form strongly penalizes cells that score poorly on any one component.

### Recommended default

Use the weighted average first because it is easier to interpret and more numerically stable.

---

# Recommended Training Weight

Convert combined confidence into a normalized training weight:

\[
w_i
=
\frac{
\left(C_i^{\mathrm{combined}}+\epsilon\right)^\gamma
}{
\mathbb{E}_i
\left[
\left(C_i^{\mathrm{combined}}+\epsilon\right)^\gamma
\right]
}.
\]

Recommended values to test:

\[
\boxed{
\gamma \in \{0.5,1.0,2.0\}
}
\]

Interpretation:

- \(\gamma=0.5\): mild reweighting;
- \(\gamma=1.0\): linear reweighting;
- \(\gamma=2.0\): strong suppression of uncertain pairs.

### Recommended starting value

\[
\boxed{\gamma=1.0}
\]

---

# Recommended Experimental Design

## Baselines

Compare at least:

1. no confidence weighting;
2. maximum-coupling confidence;
3. entropy confidence;
4. cost confidence;
5. stability confidence;
6. combined confidence.

## Artificial Noise Evaluation

Corrupt a controlled fraction of OT pairs:

\[
0\%,\ 10\%,\ 20\%,\ 30\%,\ 40\%.
\]

Evaluate whether each confidence metric assigns lower scores to corrupted pairs.

## Evaluation Metrics

Recommended evaluation measures include:

- AUROC for identifying corrupted pairs;
- AUPRC for identifying corrupted pairs;
- Spearman correlation with downstream prediction error;
- calibration curve;
- expected calibration error;
- downstream validation loss;
- MMD or Wasserstein distance;
- cell-type consistency;
- neighborhood preservation.

## Recommended Initial Configuration

\[
\boxed{
R=20,\quad
k=5,\quad
\tau=\operatorname{median}(\bar{c}),\quad
\gamma=1.0
}
\]

Combined confidence:

\[
\boxed{
C_i^{\mathrm{combined}}
=
0.35C_i^{\mathrm{entropy}}
+
0.25C_i^{\mathrm{cost}}
+
0.40C_i^{\mathrm{stability}}
}
\]

These values are reasonable starting points, not universal optimal settings. They should be tuned on validation data.

---

# Practical Recommendation

For the first version of the project:

1. Use maximum coupling as the paper baseline.
2. Add entropy and top-1/top-2 margin as low-cost alternatives.
3. Add expected transport cost to detect concentrated but biologically distant matches.
4. Run 20 bootstrap OT repetitions for stability.
5. Compare individual metrics before building a combined score.
6. Use the combined score only after confirming that the component metrics provide complementary information.

The main research question can be framed as:

> Can multi-metric confidence estimation improve the detection of unreliable OT matches and improve downstream cell-transition modeling compared with maximum coupling alone?

---

# OT Confidence Metrics: Practical Recommended Values

## Important Note

The equations define each confidence metric. The numerical values below are **recommended starting settings**, not universally optimal values. They should be validated on the target dataset.

---

## 1. Recommended Metrics for the First Experiment

Use these five metrics first:

| Metric | Recommended use | Output range |
|---|---|---:|
| Maximum coupling | SPATIA baseline | \(0\)–\(1\) |
| Entropy confidence | Main concentration metric | \(0\)–\(1\) |
| Top-1/Top-2 margin | Detect ambiguous top matches | \(0\)–\(1\) |
| Cost confidence | Detect biologically distant matches | \(0\)–\(1\) |
| Stability confidence | Measure robustness across OT runs | \(0\)–\(1\) |

The recommended primary set is:

```text
max_coupling
entropy_confidence
top12_margin
cost_confidence
stability_confidence
```

---

## 2. Numerical Constants

Use:

```text
epsilon = 1e-8
```

This prevents division by zero and taking the logarithm of zero.

---

## 3. Maximum-Coupling Confidence

Definition:

\[
C_i^{\max}=\max_j p_{ij}
\]

Recommended interpretation:

| Maximum coupling | Recommended label |
|---:|---|
| \(\ge 0.80\) | Very high concentration |
| \(0.60\)–\(0.80\) | High concentration |
| \(0.40\)–\(0.60\) | Moderate or ambiguous |
| \(0.20\)–\(0.40\) | Low concentration |
| \(<0.20\) | Very diffuse |

Recommended initial filtering threshold:

```text
max_coupling_threshold = 0.40
```

However, do not permanently remove low-confidence pairs in the first experiment. Compare filtering with continuous weighting.

---

## 4. Entropy-Based Confidence

Definition:

\[
C_i^{\mathrm{entropy}}
=
1-
\frac{-\sum_j p_{ij}\log(p_{ij}+\epsilon)}
{\log N_{\mathrm{tgt}}}
\]

Recommended interpretation:

| Entropy confidence | Recommended label |
|---:|---|
| \(\ge 0.75\) | High |
| \(0.50\)–\(0.75\) | Moderate |
| \(0.25\)–\(0.50\) | Low |
| \(<0.25\) | Very low |

Recommended initial filtering threshold:

```text
entropy_confidence_threshold = 0.50
```

Recommended use:

```text
primary_concentration_metric = entropy_confidence
```

Entropy confidence is recommended over maximum coupling when the entire OT row should be considered.

---

## 5. Top-1/Top-2 Margin

Definition:

\[
C_i^{\mathrm{margin}}
=
p_{i(1)}-p_{i(2)}
\]

Recommended interpretation:

| Margin | Recommended label |
|---:|---|
| \(\ge 0.30\) | Clearly dominant target |
| \(0.15\)–\(0.30\) | Moderate separation |
| \(0.05\)–\(0.15\) | Weak separation |
| \(<0.05\) | Almost tied |

Recommended initial threshold:

```text
top12_margin_threshold = 0.10
```

A pair with a margin below \(0.10\) should be treated as potentially ambiguous.

---

## 6. Top-1/Top-2 Ratio

Definition:

\[
C_i^{\mathrm{ratio}}
=
\frac{p_{i(1)}}{p_{i(2)}+\epsilon}
\]

Recommended interpretation:

| Ratio | Recommended label |
|---:|---|
| \(\ge 3.0\) | Strong dominance |
| \(2.0\)–\(3.0\) | Moderate dominance |
| \(1.2\)–\(2.0\) | Weak dominance |
| \(<1.2\) | Nearly tied |

Recommended initial threshold:

```text
top12_ratio_threshold = 2.0
```

For modeling, use:

```text
ratio_feature = log(top1 / (top2 + 1e-8))
```

Do not combine both raw ratio and log ratio in the same simple model unless multicollinearity is handled.

---

## 7. Effective Number of Targets

Definition:

\[
N_{\mathrm{eff},i}
=
\exp(H_i)
\]

Recommended interpretation depends on the number of target cells. For easier comparison, report the normalized effective fraction:

\[
F_{\mathrm{eff},i}
=
\frac{N_{\mathrm{eff},i}}{N_{\mathrm{tgt}}}
\]

Recommended interpretation:

| Effective target fraction | Recommended label |
|---:|---|
| \(\le 0.01\) | Highly concentrated |
| \(0.01\)–\(0.05\) | Moderately concentrated |
| \(0.05\)–\(0.20\) | Diffuse |
| \(>0.20\) | Very diffuse |

Recommended initial threshold:

```text
effective_target_fraction_threshold = 0.05
```

Because this metric is highly dependent on \(N_{\mathrm{tgt}}\), it is better used as an analysis metric than as the primary training weight.

---

## 8. Transport-Cost Confidence

Definition:

\[
\bar c_i=\sum_j p_{ij}c_{ij}
\]

\[
C_i^{\mathrm{cost}}
=
\exp\left(-\frac{\bar c_i}{\tau}\right)
\]

Recommended scale:

```text
tau = median(expected_transport_cost)
```

Recommended sensitivity analysis:

```text
tau_values = [
    0.5 * median_cost,
    1.0 * median_cost,
    2.0 * median_cost
]
```

Recommended default:

```text
tau_multiplier = 1.0
```

Recommended interpretation after conversion to \([0,1]\):

| Cost confidence | Recommended label |
|---:|---|
| \(\ge 0.70\) | Low-cost match |
| \(0.40\)–\(0.70\) | Moderate-cost match |
| \(<0.40\) | High-cost match |

Recommended initial threshold:

```text
cost_confidence_threshold = 0.40
```

A percentile-based alternative is often more robust:

```text
high_cost_pair = expected_cost > 75th_percentile
```

---

## 9. Stability Confidence

Repeat OT after perturbing the data or hyperparameters.

Recommended number of repeated runs:

```text
bootstrap_runs_initial = 20
bootstrap_runs_final = 50
```

Recommended perturbations:

```text
random_seeds = 5
gene_subsets = 4
total_runs = 20
```

For example:

- 5 random seeds
- 4 gene-subset bootstrap configurations
- total of 20 OT solutions

Recommended neighborhood size:

```text
target_neighborhood_k = 5
```

For noisier datasets:

```text
target_neighborhood_k = 10
```

Recommended interpretation:

| Stability confidence | Recommended label |
|---:|---|
| \(\ge 0.80\) | Highly stable |
| \(0.60\)–\(0.80\) | Moderately stable |
| \(0.40\)–\(0.60\) | Uncertain |
| \(<0.40\) | Unstable |

Recommended initial threshold:

```text
stability_confidence_threshold = 0.60
```

---

## 10. Cycle Consistency

Recommended neighborhood sizes:

```text
cycle_k_values = [1, 5, 10]
cycle_k_default = 5
```

Interpretation:

```text
cycle_consistent = reverse_match_rank <= 5
```

Recommended use:

- Use \(k=1\) for strict one-to-one matching.
- Use \(k=5\) as the main analysis.
- Use \(k=10\) for noisy or many-to-many transitions.

Do not use cycle consistency as the only confidence metric when the biological transition is inherently many-to-many.

---

## 11. Unbalanced-OT Retained Mass

Definition:

\[
C_i^{\mathrm{mass}}
=
\frac{\sum_jP_{ij}}{a_i+\epsilon}
\]

Recommended interpretation:

| Retained mass | Recommended label |
|---:|---|
| \(\ge 0.90\) | Strong counterpart |
| \(0.70\)–\(0.90\) | Moderate counterpart |
| \(0.50\)–\(0.70\) | Weak counterpart |
| \(<0.50\) | Potentially unmatched |

Recommended initial threshold:

```text
retained_mass_threshold = 0.70
```

Only use this metric with unbalanced OT.

---

# 12. Recommended Combined Confidence

Before combination, transform every metric to \([0,1]\).

Recommended normalization:

```text
normalization = percentile_rank
```

A practical weighted confidence score is:

\[
C_i^{\mathrm{combined}}
=
0.30C_i^{\mathrm{entropy}}
+
0.20C_i^{\mathrm{margin}}
+
0.20C_i^{\mathrm{cost}}
+
0.30C_i^{\mathrm{stability}}
\]

Recommended initial weights:

```text
entropy_weight   = 0.30
margin_weight    = 0.20
cost_weight      = 0.20
stability_weight = 0.30
```

Why these values:

- entropy measures overall concentration;
- margin catches top-target ambiguity;
- cost checks biological or geometric plausibility;
- stability checks robustness;
- entropy and stability receive slightly higher weights.

Do not include maximum coupling in this combined score initially because it is strongly correlated with entropy and margin. Keep maximum coupling as the baseline.

Recommended combined-confidence interpretation:

| Combined confidence | Recommended label |
|---:|---|
| \(\ge 0.75\) | High confidence |
| \(0.50\)–\(0.75\) | Moderate confidence |
| \(0.25\)–\(0.50\) | Low confidence |
| \(<0.25\) | Very low confidence |

Recommended threshold:

```text
combined_confidence_threshold = 0.50
```

---

# 13. Recommended Training Weights

Convert confidence to a training weight:

\[
w_i
=
\frac{(C_i+\epsilon)^\gamma}
{\operatorname{mean}[(C_i+\epsilon)^\gamma]}
\]

Recommended values:

```text
gamma_values = [0.5, 1.0, 2.0]
gamma_default = 1.0
```

Interpretation:

| \(\gamma\) | Effect |
|---:|---|
| \(0.5\) | Mild downweighting |
| \(1.0\) | Linear weighting |
| \(2.0\) | Strong downweighting |

Recommended clipping:

```text
minimum_training_weight = 0.20
maximum_training_weight = 3.00
```

The clipping prevents a small number of pairs from receiving almost zero or extremely large weights.

Recommended default weighting setup:

```text
gamma = 1.0
weight_clip = [0.2, 3.0]
```

---

# 14. Recommended OT Parameter Sensitivity Tests

Confidence depends strongly on entropy regularization. Therefore, do not evaluate confidence using only one OT regularization value.

Let the current entropy regularization be \(\varepsilon_{\mathrm{OT}}\). Test:

```text
epsilon_OT_multipliers = [0.5, 1.0, 2.0]
```

For example, if the current value is:

```text
epsilon_OT = 5e-4
```

test:

```text
epsilon_OT_values = [2.5e-4, 5e-4, 1e-3]
```

For a wider sensitivity analysis:

```text
epsilon_OT_values = [1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
```

The exact scale depends on how the transport cost matrix is normalized.

Recommended cost normalization:

```text
cost_matrix = cost_matrix / median(cost_matrix)
```

This makes OT regularization values more comparable across datasets.

---

# 15. Recommended Artificial Pairing-Noise Experiment

Corrupt the matched target for a controlled fraction of source cells.

Recommended noise levels:

```text
pairing_noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
```

Minimum initial experiment:

```text
pairing_noise_levels = [0.0, 0.1, 0.2]
```

Recommended evaluation:

```text
primary_metric = AUPRC
secondary_metrics = [AUROC, Spearman_correlation]
```

Why AUPRC is primary:

Incorrect pairs may represent a minority of all pairs, making AUPRC more informative than accuracy.

---

# 16. Recommended Initial Configuration

```yaml
epsilon_numerical: 1.0e-8

metrics:
  - maximum_coupling
  - entropy_confidence
  - top12_margin
  - cost_confidence
  - stability_confidence

thresholds:
  maximum_coupling: 0.40
  entropy_confidence: 0.50
  top12_margin: 0.10
  cost_confidence: 0.40
  stability_confidence: 0.60
  combined_confidence: 0.50

stability:
  initial_runs: 20
  final_runs: 50
  neighborhood_k: 5

cost_confidence:
  tau: median_expected_cost

combined_confidence:
  entropy_weight: 0.30
  margin_weight: 0.20
  cost_weight: 0.20
  stability_weight: 0.30

training_weight:
  gamma: 1.0
  minimum: 0.20
  maximum: 3.00

pairing_noise_levels:
  - 0.0
  - 0.1
  - 0.2
  - 0.3
  - 0.4

ot_sensitivity:
  epsilon_multipliers:
    - 0.5
    - 1.0
    - 2.0
```

---

# 17. Most Important Recommendation

Do not claim that these thresholds represent calibrated probabilities.

For the first project stage:

1. Compute all metrics without filtering.
2. Test whether they identify artificially corrupted pairs.
3. Compare their correlation with downstream prediction error.
4. Tune thresholds on validation data.
5. Only then use them for filtering or training reweighting.

The strongest initial comparison is:

```text
No confidence
vs.
Maximum coupling
vs.
Entropy confidence
vs.
Stability confidence
vs.
Combined confidence
```
