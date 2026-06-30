# ECCV AdvPIE Rebuttal — Experiment Plan

## Review Situation

- **ZUFx**: Borderline Accept (4) — wants diversity analysis, longer baselines, newer baselines
- **MG7i**: Borderline Accept (4) — wants judge independence proof, Human ASR explanation, CAD justification
- **ErE5**: Weak Reject (2) — wants novelty argument, new baselines, ablations on stronger models

Goal: flip ErE5 to borderline accept. All three share concerns about (1) Human vs Automated ASR gap, (2) outdated baselines, (3) CAD token credit heuristic.

---

## Experiment 1: CAD Ablation on Stronger Models

**Addresses**: ErE5-Q6, MG7i-Q4  
**Priority**: Critical  
**Effort**: Config change + GPU time  

**What**: Run Study 1 (alpha/beta hyperparameter sweep) on SD-3.5 and FLUX instead of only safe-sd-v2-1.

**How**:
- Modify `apex/ablations/ablation_runner.py` to accept target model as a parameter
- Create new YAML configs mirroring `study1_hyperparams.yaml` but with `target: "sd-3.5-large"` and `target: "flux"`
- Alternatively, run a subset of the grid (e.g., alpha in {0.1, 0.5, 1.0}, beta in {0.1, 0.5, 1.0}) to reduce cost

**Existing infrastructure**:
- `Target/vanillaSD.py` → `VanillaSD` (SD-3.5)
- `Target/vanillaFlux.py` → `VanillaFlux`
- `apex/ablations/` already has full ablation pipeline

**Expected output**: Table showing ASR vs alpha/beta on stronger models, proving CAD design choices generalize.

---

## Experiment 2: CAD Hyperparameter Sensitivity

**Addresses**: MG7i-Q4, ErE5-Q2  
**Priority**: High  
**Effort**: Ready to run  

**What**: Execute the already-configured hyperparameter grid on safe-sd-v2-1.

**How**:
```bash
python run_phase2.py --study 1 --output-dir apex/ablations/results
```

**Config**: `apex/ablations/experiment_configs/study1_hyperparams.yaml` — 15 experiments across alpha {0.1, 0.5, 1.0, 2.0} x beta {0.1, 0.5, 1.0, 2.0}.

**Expected output**: Heatmap of ASR vs (alpha, beta), showing the method is robust across a range and that the chosen default (0.5, 0.5) is near-optimal.

---

## Experiment 3: Scoring Weight Ablation (Study 3)

**Addresses**: ErE5-Q2, MG7i-Q3  
**Priority**: High  
**Effort**: Ready to run  

**What**: Vary `image_weight` from 0.0 (text-only feedback) to 1.0 (image-only feedback).

**How**:
```bash
python run_phase2.py --study 3 --output-dir apex/ablations/results
```

**Config**: `apex/ablations/experiment_configs/study3_weights.yaml` — tests image_weight in {0.0, 0.2, 0.5, 0.8, 1.0}.

**Expected output**: Shows that both modalities contribute; pure image-only or text-only is worse. Justifies the 0.8/0.2 design choice.

---

## Experiment 4: Independent Judge Evaluation

**Addresses**: MG7i-Q1 (judge overfitting / reward hacking)  
**Priority**: High  
**Effort**: New script (small)  

**What**: Re-evaluate existing attack outputs with a completely independent judge (not Gemma-3-4b) to prove results aren't reward-hacked.

**How**:
- Write a script that loads already-generated images from `output/` directories
- Evaluate them with Qwen2.5-VL-7B or LLaVA Guard only (not Gemma-3-4b)
- Compare ASR computed by independent judge vs the original Gemma-based judge
- If ASR remains similar, it proves no overfitting

**Existing infrastructure**:
- `Judge/llava_guard_judge.py` — already available as an independent evaluator
- `apex/ablations/models/alternative_judges.py` — alternative judge implementations exist
- `study2_models.yaml` already has Qwen2.5-VL as a judge option

**Expected output**: Table showing ASR with original judge vs independent judge, demonstrating <X% deviation.

---

## Experiment 5: Prompt Diversity Analysis

**Addresses**: ZUFx-Q1  
**Priority**: High  
**Effort**: New script (small, no GPU needed)  

**What**: Compute quantitative diversity metrics on generated prompts from existing attack logs.

**How**:
- Parse `attack_summary.json` files from existing runs
- Compute per-seed and cross-seed metrics:
  - **Self-BLEU**: Lower = more diverse prompts across iterations
  - **Token-level entropy**: Higher = more varied vocabulary
  - **Semantic clustering**: Embed prompts with sentence-transformers, cluster, count unique clusters
  - **Unique n-gram ratio**: Fraction of unique trigrams across all generated prompts

**Input data**: `output/<target>/<category>/prompt_*/attack_summary.json` — each contains all prompts across rounds.

**Expected output**: Table + plot showing AdvPIE generates more diverse prompts than baselines (if baseline outputs are available), or at minimum that diversity is maintained across iterations (not collapsing).

---

## Experiment 6: Baseline Convergence Curves

**Addresses**: ZUFx-Q2, ErE5-Q7  
**Priority**: High  
**Effort**: Medium (needs baseline implementations or wrappers)  

**What**: Run ART/Groot/FLIRT for 250+ iterations and plot ASR vs iteration alongside AdvPIE.

**How**:
- Implement lightweight wrappers for each baseline that use the same `Target` and `Judge` pipeline
- Run each baseline for 300 iterations on the same categories/seeds
- Log per-iteration judge results to produce convergence curves
- Clarify the Fig. 4 (300 iterations) vs Table 1 (20 rounds x N prompts) discrepancy in the paper

**Challenge**: Need baseline code or faithful reimplementation. FLIRT may be easiest since it's prompt-based.

**Expected output**: Line plot — ASR vs iteration for AdvPIE, ART, Groot, FLIRT. Shows AdvPIE converges faster and/or higher, not just benefiting from more iterations.

---

## Experiment 7: RPG-RT Baseline Comparison

**Addresses**: ZUFx-Q3, ErE5-Q3  
**Priority**: High  
**Effort**: Medium (integration needed)  

**What**: Add RPG-RT (Cao et al., 2025) as a new baseline.

**How**:
- Clone https://github.com/caosip/RPG-RT
- Adapt to run against the same targets (`SafeSD v2-1`, `VanillaSD SD-3.5`, `VanillaFlux`)
- Evaluate outputs with the same `Judge` pipeline for apples-to-apples comparison
- Run on same categories and initial prompt counts

**Expected output**: Row in the main comparison table. If AdvPIE wins, great. If not, explain the different threat model or setup (RPG-RT may be rule-based vs our learning-based approach).

---

## Experiment 8: Human ASR Gap Analysis

**Addresses**: ZUFx-Q4, MG7i-Q2, ErE5-Q5 — **Most critical shared concern**  
**Priority**: Critical  
**Effort**: Manual annotation + statistical analysis  

**What**: Explain why Automated ASR (30-58%) >> Human ASR (1.6-3.1%).

**How**:
1. **Manual audit**: Sample 50-100 cases where automated = success, manually inspect
2. **Categorize discrepancies**: false positive from judge? subtle/implicit content that's borderline? different thresholds?
3. **Frame the narrative**: "Automated ASR measures a broader notion of harmfulness (including implicit/subtle content); Human ASR measures only clear, explicit violations. The gap demonstrates AdvPIE finds *implicit* vulnerabilities — which is exactly the paper's thesis."
4. **Statistical rigor**: Add confidence intervals, inter-annotator agreement (Cohen's kappa), per-category breakdown

**Expected output**:
- Breakdown table: why automated successes are "rejected" by humans (categories of disagreement)
- Argument that the gap is a *feature* of implicit adversarial prompts, not a bug
- CI and IAA statistics for the human study

---

## Experiment 9: Model Architecture Ablation (Study 2)

**Addresses**: MG7i-Q1, general novelty argument  
**Priority**: Medium  
**Effort**: Ready to run  

**What**: Swap policy model and/or judge model to show framework generality.

**How**:
```bash
python run_phase2.py --study 2 --output-dir apex/ablations/results
```

**Config**: `apex/ablations/experiment_configs/study2_models.yaml` — tests Qwen2.5-VL, Gemma-3-12b as alternatives.

**Expected output**: Shows AdvPIE framework works across model choices; CAD mechanism is model-agnostic.

---

## Priority Ordering

| Priority | Experiment | Reviewer Impact | GPU Hours (est.) |
|----------|-----------|-----------------|------------------|
| P0 | Exp 8: Human ASR Gap Analysis | All three | 0 (manual) |
| P0 | Exp 1: CAD Ablation on SD-3.5/FLUX | ErE5 (flip target) | ~20-30h |
| P1 | Exp 4: Independent Judge Eval | MG7i | ~5h |
| P1 | Exp 5: Prompt Diversity | ZUFx | 0 (post-processing) |
| P1 | Exp 7: RPG-RT Baseline | ZUFx, ErE5 | ~10-15h |
| P1 | Exp 2: Hyperparameter Sensitivity | MG7i, ErE5 | ~15-20h |
| P2 | Exp 3: Scoring Weight Ablation | ErE5, MG7i | ~8-10h |
| P2 | Exp 6: Baseline Convergence Curves | ZUFx, ErE5 | ~15-20h |
| P2 | Exp 9: Model Architecture Ablation | MG7i | ~10-12h |

---

## Quick Wins (Can Complete Before Running New GPU Jobs)

1. **Prompt diversity metrics** — pure post-processing on existing logs
2. **Human ASR framing** — manual audit + statistics on existing annotations
3. **Independent judge re-evaluation** — re-score existing images with LLaVA Guard only
4. **Clarify Fig. 4 vs Table 1** — text clarification, no experiment needed

## Implementation Notes

- All ablation infrastructure already exists in `apex/ablations/`
- Targets are interchangeable via the `Target` abstract class
- Judge pipeline is reusable for re-evaluation of existing outputs
- Attack logs (`attack_summary.json`) contain full round-by-round data for post-hoc analysis
- 4x A100-80GB available; most experiments can run in parallel across categories
