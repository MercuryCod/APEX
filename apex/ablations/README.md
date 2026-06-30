# APEX Ablation Studies Implementation

## 📋 **Phase 1: Infrastructure Setup (COMPLETED)**

This directory contains the complete implementation for conducting systematic ablation studies on the APEX red-teaming framework. All experiments use **safe-sd-v2-1** as the target model and maintain **identical baseline settings** for valid ablation analysis.

### **🎯 Key Principle: Controlled Variables**

**Critical for Valid Ablation**: All settings remain identical to the original implementation except for the specific variable being tested:

- **Baseline Settings** (identical across all studies):
  - Target: `safe-sd-v2-1` (using `SafeSD("v2-1")`)
  - Category: `sexual` (consistent across all experiments)
  - Rounds: `20` (same as main.py default)
  - Samples: `2` (same as main.py `num_sample`)
  - Initial Prompts: `30` (same as main.py `NUM_INITIAL_PROMPTS`)
  - Max Tokens: `77` (same as main.py for safe-sd targets)
  - Devices: External Judge `["cuda:1","cuda:2"]`, APEX `["cuda:3","cuda:4"]`
  - All prompts, templates, and other parameters unchanged

### **📁 Directory Structure**

```
apex/ablations/
├── __init__.py                    # Module exports
├── configurable_apex.py           # Extended APEX with configurable parameters
├── ablation_runner.py             # Experiment orchestration
├── test_phase1.py                 # Infrastructure validation tests
├── README.md                      # This documentation
│
├── experiment_configs/            # YAML experiment configurations
│   ├── study1_hyperparams.yaml   # Alpha/beta parameter sweep
│   ├── study2_models.yaml         # Model architecture variations
│   └── study3_weights.yaml        # Image/text weight balance
│
├── models/                        # Alternative model implementations
│   ├── __init__.py               # Model exports
│   └── alternative_judges.py      # Qwen/Gemma judge agents
│
├── results/                       # Experiment results (gitignored)
│   ├── study1_results/           # Hyperparameter ablation results
│   ├── study2_results/           # Model ablation results
│   └── study3_results/           # Weight ablation results
│
└── analysis/                     # Analysis scripts and notebooks
    └── (Future: metrics_calculator.py, plots, etc.)
```

### **🧪 Three Ablation Studies**

#### **Study 1: Hyperparameter Ablations (15 experiments)**
- **Variable**: Alpha (learning rate) and Beta (decay factor) of `AdatptiveLogitsProcessor`
- **Fixed**: All models, weights, and other settings identical to baseline
- **Grid**: α ∈ {0.1, 0.5, 1.0, 2.0} × β ∈ {0.1, 0.5, 1.0, 2.0} (excluding baseline α=0.5, β=0.5)

#### **Study 2: Model Architecture Ablations (5 experiments)**
- **Variable**: Policy generation model and/or internal judge model
- **Fixed**: Hyperparameters (α=0.5, β=0.5, image_weight=0.8), all other settings
- **Models**: LLaVA-1.6-Mistral-7B, Qwen2.5-VL-7B, Gemma-3-4B, Gemma-3-12B

#### **Study 3: Scoring Weight Ablations (4 experiments)**
- **Variable**: Image vs text weight balance in score calculation
- **Fixed**: Models (LLaVA + Gemma-3-4B), hyperparameters (α=0.5, β=0.5), all other settings  
- **Weights**: image_weight ∈ {0.0, 0.2, 0.5, 1.0} (excluding baseline 0.8)

### **🔧 Key Components**

#### **ConfigurableAPEX**
Extends the original APEX class with configurable parameters while preserving all baseline behavior:
```python
from apex.ablations import ConfigurableAPEX

# Hyperparameter ablation example
apex = ConfigurableAPEX(
    max_new_tokens=77,
    devices=["cuda:3", "cuda:4"],
    alpha=1.0,              # Variable: test different alpha
    beta=2.0,               # Variable: test different beta
    image_weight=0.8,       # Fixed: baseline weight
    policy_model="llava-1.6-mistral-7b",  # Fixed: baseline model
    judge_model="gemma-3-4b"              # Fixed: baseline model
)
```

#### **AblationRunner**
Orchestrates systematic experiment execution with automatic skipping of completed runs:
```python
from apex.ablations import AblationRunner

runner = AblationRunner(base_output_dir="apex/ablations/results")

# Run individual studies
runner.run_study_1_hyperparams()
runner.run_study_2_models()  
runner.run_study_3_weights()

# Or run all studies
runner.run_all_studies()
```

### **🚀 Usage**

#### **Prerequisites** 
```bash
# Ensure environment is set up
conda activate apex  # or your environment name

# Install additional dependencies (if needed)
pip install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils[decord]==0.0.8

# Verify GPU availability (check for conflicts with other users)
nvidia-smi
```

#### **Run Infrastructure Tests**
```bash
python apex/ablations/test_phase1.py
```

#### **Execute Ablation Studies**
```python
from apex.ablations import AblationRunner

# Initialize runner
runner = AblationRunner()

# Run specific study
runner.run_study_1_hyperparams()  # ~15-20 hours
runner.run_study_2_models()       # ~10-12 hours  
runner.run_study_3_weights()      # ~8-10 hours

# Or run all studies in sequence
runner.run_all_studies()          # ~35-40 hours total
```

### **📊 Expected Outputs**

Each experiment generates:
- **Results Directory**: `apex/ablations/results/study{N}_results/baseline_{config}/`
- **Experiment Log**: JSON file with configuration, timing, and results summary
- **Attack Results**: Same format as original APEX (images, prompts, analysis)
- **Progress Tracking**: Automatic skip of completed experiments for resume capability

### **🔒 Git Integration**

- **Open Source Ready**: `apex/ablations/` is gitignored, keeping main implementation clean
- **Original Code Unchanged**: All original APEX code preserved for release
- **Clean Separation**: Ablation experiments don't affect production codebase

### **✅ Phase 1 Status: COMPLETE**

All infrastructure components implemented and tested:
- ✅ Directory structure created
- ✅ ConfigurableAPEX class implemented  
- ✅ AblationRunner orchestration ready
- ✅ Experiment configurations defined
- ✅ Alternative model implementations created
- ✅ Baseline consistency ensured
- ✅ Git integration configured

**Next**: Ready for Phase 2 - Execute Study 1 hyperparameter ablations.

---

*Last Updated: 2025-01-23*
*Implementation: Phase 1 Complete - Infrastructure Ready for Ablation Experiments* 