"""
Ablation study runner for APEX experiments.
Orchestrates systematic testing of hyperparameters, models, and weights.
Target model is fixed to safe-sd-v2-1 for all experiments.
"""

import os
import yaml
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .configurable_apex import ConfigurableAPEX
from Target import VanillaSD  # Assuming safe-sd-v2-1 uses VanillaSD class
from Judge import Judge
from Runner import ApexRunner
from harmful_content import HarmfulContentManager

class AblationRunner:
    """
    Orchestrates ablation study experiments across different configurations.
    Handles experiment execution, result logging, and progress tracking.
    """
    
    def __init__(self, base_output_dir: str = "apex/ablations/results"):
        """
        Initialize ablation runner.
        
        Args:
            base_output_dir: Base directory for storing results
        """
        self.base_output_dir = base_output_dir
        self.content_manager = HarmfulContentManager()
        
        # Ensure output directories exist
        os.makedirs(base_output_dir, exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def should_skip_experiment(self, exp_config: Dict[str, Any], study_dir: str) -> bool:
        """Check if experiment already completed to avoid reruns."""
        # Create expected output folder name based on experiment type
        if 'alpha' in exp_config and 'beta' in exp_config:
            folder_name = f"baseline_alpha{exp_config['alpha']}_beta{exp_config['beta']}"
        elif 'policy_model' in exp_config and 'judge_model' in exp_config:
            folder_name = f"baseline_{exp_config['name']}"
        elif 'image_weight' in exp_config:
            folder_name = f"baseline_{exp_config['name']}"
        else:
            folder_name = exp_config.get('name', 'unknown')
            
        output_folder = os.path.join(study_dir, folder_name)
        
        # Check if results already exist and are non-empty
        if os.path.exists(output_folder):
            contents = os.listdir(output_folder)
            if len(contents) > 0:
                print(f"Skipping {folder_name} - results already exist")
                return True
        return False
    
    def create_experiment_log(self, config: Dict[str, Any], study_dir: str, exp_name: str) -> str:
        """Create experiment log file with configuration details."""
        log_file = os.path.join(study_dir, f"{exp_name}_experiment_log.json")
        
        log_data = {
            "experiment_name": exp_name,
            "start_time": datetime.now().isoformat(),
            "configuration": config,
            "target_model": "safe-sd-v2-1",
            "status": "running"
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        return log_file
    
    def update_experiment_log(self, log_file: str, status: str, results: Dict = None):
        """Update experiment log with completion status and results."""
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        log_data["status"] = status
        log_data["end_time"] = datetime.now().isoformat()
        if results:
            log_data["results_summary"] = results
            
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def run_single_experiment(self, config: Dict[str, Any], output_folder: str, exp_name: str) -> Dict[str, Any]:
        """
        Run a single ablation experiment with given configuration.
        
        Args:
            config: Experiment configuration
            output_folder: Directory to save results
            exp_name: Name of the experiment for logging
            
        Returns:
            Dictionary with experiment results summary
        """
        print(f"\n{'='*60}")
        print(f"Starting experiment: {exp_name}")
        print(f"Configuration: {config}")
        print(f"Output folder: {output_folder}")
        print(f"Target: safe-sd-v2-1 (fixed)")
        print(f"{'='*60}")
        
        try:
            # Initialize target model (safe-sd-v2-1) - EXACTLY as in main.py
            from Target import SafeSD
            target = SafeSD("v2-1")  # Exact same as main.py for safe-sd-v2-1
            
            # Initialize external judge - Using available GPUs (avoiding occupied 0,1)
            external_judge = Judge(devices=["cuda:2", "cuda:3"])  # Use available GPUs
            
            # Initialize configurable APEX with experiment parameters - Using available GPUs
            apex_method = ConfigurableAPEX(
                max_new_tokens=config.get('max_new_tokens', 77),
                devices=["cuda:4", "cuda:5"],  # Use available GPUs (fallback to CPU if not available)
                alpha=config.get('alpha', 0.5),
                beta=config.get('beta', 0.5),
                image_weight=config.get('image_weight', 0.8),
                policy_model=config.get('policy_model', 'llava-1.6-mistral-7b'),
                judge_model=config.get('judge_model', 'gemma-3-4b')
            )
            
            # Create orchestrator - EXACTLY as in main.py
            attacker = ApexRunner(
                target=target,
                judge=external_judge,
                apex=apex_method,
                num_sample=config.get('num_samples', 2)  # Same default as main.py
            )
            
            # Get initial prompts for the category - EXACTLY as in main.py
            category = config.get('category', 'sexual')
            initial_prompts = self.content_manager.get_initial_prompts(
                category, config.get('num_initial_prompts', 30)  # Same default as main.py
            )
            
            # Run attack for each initial prompt
            results_summary = {
                "total_prompts": len(initial_prompts),
                "successful_attacks": 0,
                "failed_attacks": 0,
                "total_rounds": 0,
                "avg_rounds_to_success": 0
            }
            
            for i, prompt in enumerate(initial_prompts):
                prompt_output_dir = os.path.join(output_folder, f"prompt_{i+1}")
                os.makedirs(prompt_output_dir, exist_ok=True)
                
                try:
                    print(f"\nRunning attack {i+1}/{len(initial_prompts)} with prompt: {prompt[:50]}...")
                    
                    # Run attack
                    attacker.attack_by_category(
                        initial_prompt=prompt,
                        output_folder=prompt_output_dir,
                        category_name=category,
                        max_round=config.get('num_rounds', 20)
                    )
                    
                    results_summary["successful_attacks"] += 1
                    print(f"Attack {i+1} completed successfully")
                    
                except Exception as e:
                    print(f"Attack {i+1} failed: {str(e)}")
                    results_summary["failed_attacks"] += 1
                    continue
            
            # Calculate summary statistics
            if results_summary["successful_attacks"] > 0:
                success_rate = results_summary["successful_attacks"] / results_summary["total_prompts"]
                results_summary["success_rate"] = success_rate
                print(f"\nExperiment {exp_name} completed!")
                print(f"Success rate: {success_rate:.2%}")
            else:
                results_summary["success_rate"] = 0.0
                print(f"\nExperiment {exp_name} completed with no successful attacks")
                
            return results_summary
            
        except Exception as e:
            error_msg = f"Experiment {exp_name} failed with error: {str(e)}"
            print(f"\nERROR: {error_msg}")
            return {"error": error_msg, "success_rate": 0.0}
    
    def run_study_1_hyperparams(self, config_path: str = "apex/ablations/experiment_configs/study1_hyperparams.yaml"):
        """Execute Study 1: Hyperparameter ablations (alpha, beta)."""
        print("\n🎯 Starting Study 1: Hyperparameter Ablations")
        config = self.load_config(config_path)
        study_dir = os.path.join(self.base_output_dir, "study1_results")
        os.makedirs(study_dir, exist_ok=True)
        
        base_config = config['base_config']
        
        for exp_config in config['new_experiments']:
            if self.should_skip_experiment(exp_config, study_dir):
                continue
            
            # Merge base config with experiment-specific parameters
            full_config = {**base_config, **exp_config}
            
            # Create output folder
            folder_name = f"baseline_alpha{exp_config['alpha']}_beta{exp_config['beta']}"
            output_folder = os.path.join(study_dir, folder_name)
            
            # Create experiment log
            log_file = self.create_experiment_log(full_config, study_dir, folder_name)
            
            try:
                # Run experiment
                results = self.run_single_experiment(full_config, output_folder, folder_name)
                self.update_experiment_log(log_file, "completed", results)
                
            except Exception as e:
                error_results = {"error": str(e)}
                self.update_experiment_log(log_file, "failed", error_results)
                print(f"Study 1 experiment {folder_name} failed: {str(e)}")
                continue
        
        print("\n✅ Study 1: Hyperparameter Ablations completed!")
    
    def run_study_2_models(self, config_path: str = "apex/ablations/experiment_configs/study2_models.yaml"):
        """Execute Study 2: Model architecture ablations."""
        print("\n🎯 Starting Study 2: Model Architecture Ablations")
        config = self.load_config(config_path)
        study_dir = os.path.join(self.base_output_dir, "study2_results")
        os.makedirs(study_dir, exist_ok=True)
        
        base_config = config['base_config']
        
        for exp_config in config['new_experiments']:
            if self.should_skip_experiment(exp_config, study_dir):
                continue
            
            # Merge base config with experiment-specific parameters
            full_config = {**base_config, **exp_config}
            
            # Create output folder
            folder_name = f"baseline_{exp_config['name']}"
            output_folder = os.path.join(study_dir, folder_name)
            
            # Create experiment log
            log_file = self.create_experiment_log(full_config, study_dir, folder_name)
            
            try:
                # Run experiment
                results = self.run_single_experiment(full_config, output_folder, folder_name)
                self.update_experiment_log(log_file, "completed", results)
                
            except Exception as e:
                error_results = {"error": str(e)}
                self.update_experiment_log(log_file, "failed", error_results)
                print(f"Study 2 experiment {folder_name} failed: {str(e)}")
                continue
        
        print("\n✅ Study 2: Model Architecture Ablations completed!")
    
    def run_study_3_weights(self, config_path: str = "apex/ablations/experiment_configs/study3_weights.yaml"):
        """Execute Study 3: Scoring weight ablations."""
        print("\n🎯 Starting Study 3: Scoring Weight Ablations")
        config = self.load_config(config_path)
        study_dir = os.path.join(self.base_output_dir, "study3_results")
        os.makedirs(study_dir, exist_ok=True)
        
        base_config = config['base_config']
        
        for exp_config in config['new_experiments']:
            if self.should_skip_experiment(exp_config, study_dir):
                continue
            
            # Merge base config with experiment-specific parameters
            full_config = {**base_config, **exp_config}
            
            # Create output folder
            folder_name = f"baseline_{exp_config['name']}"
            output_folder = os.path.join(study_dir, folder_name)
            
            # Create experiment log
            log_file = self.create_experiment_log(full_config, study_dir, folder_name)
            
            try:
                # Run experiment
                results = self.run_single_experiment(full_config, output_folder, folder_name)
                self.update_experiment_log(log_file, "completed", results)
                
            except Exception as e:
                error_results = {"error": str(e)}
                self.update_experiment_log(log_file, "failed", error_results)
                print(f"Study 3 experiment {folder_name} failed: {str(e)}")
                continue
        
        print("\n✅ Study 3: Scoring Weight Ablations completed!")
    
    def run_all_studies(self):
        """Run all three ablation studies in sequence."""
        print("\n🚀 Starting Complete APEX Ablation Study Suite")
        print("Target Model: safe-sd-v2-1 (fixed for all experiments)")
        print("Category: sexual (consistent across all studies)")
        
        start_time = time.time()
        
        try:
            self.run_study_1_hyperparams()
            self.run_study_2_models()
            self.run_study_3_weights()
            
            end_time = time.time()
            duration = (end_time - start_time) / 3600  # Convert to hours
            
            print(f"\n🎉 All APEX ablation studies completed!")
            print(f"Total duration: {duration:.2f} hours")
            print(f"Results saved in: {self.base_output_dir}")
            
        except Exception as e:
            print(f"\n❌ Ablation study suite failed: {str(e)}")
            raise 