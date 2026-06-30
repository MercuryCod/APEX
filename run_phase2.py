#!/usr/bin/env python3
"""
APEX Phase 2: Ablation Studies Execution Script

This script executes the systematic ablation studies on APEX components:
- Study 1: Hyperparameter ablations (α=0.1-2.0, β=0.1-2.0) - 15 experiments
- Study 2: Model architecture ablations (LLaVA, Qwen, Gemma variations) - 5 experiments  
- Study 3: Scoring weight ablations (image_weight=0.0-1.0) - 4 experiments

Total: 24 experiments, ~35-40 hours on multi-GPU setup
All experiments use safe-sd-v2-1 target with identical baseline settings.
"""

import argparse
import sys
import time
from datetime import datetime

from apex.ablations import AblationRunner

def print_header():
    """Print Phase 2 execution header."""
    print("🚀" + "="*78 + "🚀")
    print("   APEX PHASE 2: ABLATION STUDIES EXECUTION")
    print("🚀" + "="*78 + "🚀")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Target: safe-sd-v2-1 (fixed across all studies)")
    print("Category: sexual (consistent for valid comparison)")
    print("Total Experiments: 24 (15 + 5 + 4)")
    print("Estimated Duration: 35-40 hours")
    print("="*80)

def check_system_readiness():
    """Check system readiness for ablation studies."""
    print("\n🔧 SYSTEM READINESS CHECK")
    print("="*50)
    
    # Check GPU availability
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {gpu_count}")
    
    if gpu_count < 4:
        print("⚠️  WARNING: Recommended 4+ GPUs for optimal performance")
    
    # Check memory for each GPU
    for i in range(min(gpu_count, 4)):
        try:
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {mem_free/1024**3:.1f}GB free / {mem_total/1024**3:.1f}GB total")
        except:
            print(f"GPU {i}: Unable to check memory")
    
    print("✅ System check complete")
    return True

def execute_study_1(runner, dry_run=False):
    """Execute Study 1: Hyperparameter Ablations."""
    print("\n🎯 STUDY 1: HYPERPARAMETER ABLATIONS")
    print("="*50)
    print("Focus: Alpha (learning rate) and Beta (decay factor)")
    print("Grid: α ∈ {0.1, 0.5, 1.0, 2.0} × β ∈ {0.1, 0.5, 1.0, 2.0}")
    print("Experiments: 15 (excluding baseline α=0.5, β=0.5)")
    print("Estimated: 15-20 hours")
    
    if dry_run:
        print("🏃 DRY RUN: Hyperparameter ablations configured")
        return 0
    
    print("▶️  Starting hyperparameter ablations...")
    start = time.time()
    try:
        runner.run_study_1_hyperparams()
        duration = (time.time() - start) / 3600
        print(f"✅ Study 1 completed in {duration:.2f} hours")
        return duration
    except Exception as e:
        print(f"❌ Study 1 failed: {e}")
        raise

def execute_study_2(runner, dry_run=False):
    """Execute Study 2: Model Architecture Ablations."""
    print("\n🎯 STUDY 2: MODEL ARCHITECTURE ABLATIONS")
    print("="*50)
    print("Focus: Policy and judge model variations")
    print("Models: LLaVA-1.6, Qwen2.5-VL-7B, Gemma-3-4B, Gemma-3-12B")
    print("Experiments: 5 (excluding baseline)")
    print("Estimated: 10-12 hours")
    
    if dry_run:
        print("🏃 DRY RUN: Model architecture ablations configured")
        return 0
    
    print("▶️  Starting model architecture ablations...")
    start = time.time()
    try:
        runner.run_study_2_models()
        duration = (time.time() - start) / 3600
        print(f"✅ Study 2 completed in {duration:.2f} hours")
        return duration
    except Exception as e:
        print(f"❌ Study 2 failed: {e}")
        raise

def execute_study_3(runner, dry_run=False):
    """Execute Study 3: Scoring Weight Ablations."""
    print("\n🎯 STUDY 3: SCORING WEIGHT ABLATIONS")
    print("="*50)
    print("Focus: Image vs text weight balance")
    print("Weights: image_weight ∈ {0.0, 0.2, 0.5, 1.0} (baseline: 0.8)")
    print("Experiments: 4 (excluding baseline)")
    print("Estimated: 8-10 hours")
    
    if dry_run:
        print("🏃 DRY RUN: Scoring weight ablations configured")
        return 0
    
    print("▶️  Starting scoring weight ablations...")
    start = time.time()
    try:
        runner.run_study_3_weights()
        duration = (time.time() - start) / 3600
        print(f"✅ Study 3 completed in {duration:.2f} hours")
        return duration
    except Exception as e:
        print(f"❌ Study 3 failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="APEX Phase 2: Ablation Studies")
    parser.add_argument("--study", choices=["1", "2", "3", "all"], default="all",
                       help="Which study to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show configuration without running experiments")
    parser.add_argument("--output-dir", default="apex/ablations/results",
                       help="Output directory for results")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip system readiness checks")
    
    args = parser.parse_args()
    
    print_header()
    
    # System checks
    if not args.skip_checks:
        check_system_readiness()
    
    # Initialize runner
    print(f"\n📋 Initializing AblationRunner...")
    print(f"Output directory: {args.output_dir}")
    runner = AblationRunner(base_output_dir=args.output_dir)
    print("✅ AblationRunner initialized")
    
    # Execute studies
    total_start = time.time()
    total_duration = 0
    
    try:
        if args.study in ["1", "all"]:
            total_duration += execute_study_1(runner, args.dry_run)
        
        if args.study in ["2", "all"]:
            total_duration += execute_study_2(runner, args.dry_run)
        
        if args.study in ["3", "all"]:
            total_duration += execute_study_3(runner, args.dry_run)
        
        actual_duration = (time.time() - total_start) / 3600
        
        print("\n🎉" + "="*78 + "🎉")
        print("   PHASE 2 EXECUTION COMPLETE!")
        print("🎉" + "="*78 + "🎉")
        if not args.dry_run:
            print(f"Total execution time: {actual_duration:.2f} hours")
            print(f"Sum of study durations: {total_duration:.2f} hours")
        print(f"Results saved in: {args.output_dir}")
        print("Ready for Phase 3: Analysis and evaluation")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n❌ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 