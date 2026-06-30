# APEX: Adaptive Prompt Evolution with eXpert Guidance

A red-teaming framework for evaluating the robustness of text-to-image models against adversarial prompts.

> **⚠️ Ethics & Responsible Use.** APEX is a research tool for *evaluating and improving* the safety of text-to-image models. This repository contains adversarial prompt seeds and example outputs that are, by design, harmful or offensive. It is released solely to support safety research, red-teaming, and the development of stronger defenses. **Do not** use it to generate, distribute, or deploy harmful content, to attack systems you do not own or have permission to test, or for any unlawful purpose. By using this code you accept responsibility for ensuring your use complies with applicable laws, model providers' terms of service, and your institution's ethical guidelines.

## Prerequisites

- Python 3.10+
- CUDA-capable GPUs (4+ GPUs recommended)
- [`uv`](https://docs.astral.sh/uv/) (for environment management)

## Setup

```bash
# Create the .venv, install PyTorch (cu124) + dependencies, and the spaCy model
bash setup.sh
source .venv/bin/activate

# Configure API keys
cp .env.example .env
# Edit .env with your keys (OPENAI_API_KEY, STABLE_DIFFUSION_API_KEY, HF_TOKEN)
```

`setup.sh` requires `uv` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`).
It pins PyTorch 2.6.0+cu124; adjust the index URL in `setup.sh` for a different CUDA version.

## Usage

### Running APEX

```bash
python main.py --target_name <target> --base_folder ./output \
    --judge_devices cuda:0 cuda:1 \
    --apex_devices cuda:2 cuda:3
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--target_name` | Target model: `sd-3.5-large`, `safe-sd-v2-1`, `safe-sd-v1-5`, `sd-api`, `flux` | (required) |
| `--base_folder` | Output directory for results | (required) |
| `--num_round` | Attack rounds per prompt | 20 |
| `--num_sample` | Images generated per round | 2 |
| `--judge_devices` | Two CUDA devices for the external judge | `cuda:2 cuda:3` |
| `--apex_devices` | Two CUDA devices for APEX (LLaVA + internal judge) | `cuda:4 cuda:5` |

Or use the convenience script:

```bash
bash run_apex.sh
```

### Running baselines

Three baseline red-teaming methods are bundled for comparison: **ART**, **Groot**, and **Flirt**. They share the same `Judge`, `Target`, and `harmful_content` infrastructure as APEX but each drives its own attack loop.

```bash
python run_baseline.py --method <art|groot|flirt> --target_name <target> \
    --base_folder ./output \
    --judge_devices cuda:0 cuda:1 \
    --method_devices cuda:2 cuda:3
```

**Arguments specific to baselines:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | Baseline method: `art`, `groot`, or `flirt` | (required) |
| `--method_devices` | CUDA devices for the baseline. ART needs 2 (Llama writer + LLaVA guide); Groot needs 1 (LLaVA); Flirt needs 2 (LLaVA + Judge_Agent for prompt scoring). | `cuda:2 cuda:3` |

All other arguments (`--target_name`, `--num_round`, `--num_sample`, `--base_folder`, `--judge_devices`) match `main.py`.

Or use the convenience script:

```bash
bash run_baseline.sh art      # or groot, or flirt
```

**Method summaries:**

- **ART** — A LoRA-finetuned LLaVA "guide" inspects the latest image and emits a rewriting instruction; a LoRA-finetuned Llama "writer" rewrites the prompt accordingly. Image-conditioned, no score feedback.
- **Groot** — Decomposes the prompt into a JSON "Prompt Parse Tree", randomly picks a leaf, asks LLaVA to redescribe it without sensitive terms, then reconstructs the prompt. Open-loop (image is generated for evaluation but not fed back).
- **Flirt** — Maintains a queue of 5 example prompts; each round LLaVA generates a new prompt conditioned on the queue, and the lowest-scoring entry is replaced if the new prompt scores higher. Prompt-level scores come from APEX's `Judge_Agent` (Gemma).

## Output

Results are saved to `<base_folder>/<target_name>/<...>/prompt_<n>/`:

- **APEX** writes to `<base_folder>/<target_name>/<category>/prompt_<n>/`:
  - `round_<n>/` — Generated images per round
  - `attack_summary.json` — Full attack log with prompts, scores, and judge reports
- **Baselines** write to `<base_folder>/<target_name>/<method>/<category>/prompt_<n>/`:
  - `round_<n>/` — Generated images per round
  - `results.json` — Per-round log (prompt, image path, judge report)

## Project Structure

```
apex/             - Core APEX method (prompt generation, logits processing, internal judge)
baselines/        - Baseline red-teaming methods (ART, Groot, Flirt)
  base_attacker.py  - Shared abstract base class for baselines
  art/, groot/, flirt/  - One subpackage per method
Judge/            - External judge models (Gemma, LLaVA Guard, OpenAI moderation)
Target/           - Target model wrappers (SD 3.5, SafeSD, Flux, Stability API)
Runner/           - APEX attack orchestration
harmful_content/  - Category definitions and initial prompts
utils/            - Shared utilities (JSON parsing, image loading, model helpers)
main.py           - APEX entry point
run_baseline.py   - Baseline entry point
```

## Citation

If you use APEX in your research, please cite the paper:

```bibtex
@inproceedings{apex,
  title     = {APEX: Adaptive Prompt Evolution with eXpert Guidance},
  author    = {<authors>},
  booktitle = {<venue>},
  year      = {<year>}
}
```

<!-- TODO: fill in author/venue/year and link to the arXiv/DOI once published. -->
