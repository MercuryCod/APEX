from Target import SDAPI, SafeSD, VanillaSD, VanillaFlux
from Judge import Judge
from Runner import ApexRunner
from apex import APEX
from harmful_content import HarmfulContentManager
import os
import torch
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='APEX Red-Team Framework')
    parser.add_argument('--target_name', type=str, required=True,
                       choices=["sd-3.5-large", "safe-sd-v2-1", "safe-sd-v1-5", "sd-api", "flux"],
                       help='Target model to attack')
    parser.add_argument('--num_round', type=int, default=20,
                       help='Number of attack rounds per prompt')
    parser.add_argument('--num_sample', type=int, default=2,
                       help='Number of samples to generate per round')
    parser.add_argument('--base_folder', type=str, required=True,
                       help='Base output folder for results')
    parser.add_argument('--judge_devices', type=str, nargs=2, default=["cuda:2", "cuda:3"],
                       help='Devices for external judge (default: cuda:2 cuda:3)')
    parser.add_argument('--apex_devices', type=str, nargs=2, default=["cuda:4", "cuda:5"], 
                       help='Devices for APEX method (default: cuda:4 cuda:5)')
    
    args = parser.parse_args()
    
    target_name = args.target_name
    num_round = args.num_round
    num_sample = args.num_sample
    base_folder = args.base_folder
    judge_devices = args.judge_devices
    apex_devices = args.apex_devices
    
    
    # Add proper CUDA memory cleanup at start
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    NUM_INITIAL_PROMPTS = 30
    content_manager = HarmfulContentManager()
    category_names = content_manager.get_all_category_names()

    try:
        if target_name == "sd-3.5-large":
            target = VanillaSD()
            max_new_tokens = 77
        elif target_name == "safe-sd-v2-1":
            target = SafeSD("v2-1")
            max_new_tokens = 77
        elif target_name == "safe-sd-v1-5":
            target = SafeSD("v1-5")
            max_new_tokens = 77
        elif target_name == "flux":
            target = VanillaFlux()
            max_new_tokens = 77
        elif target_name == "sd-api":
            target = SDAPI()
            max_new_tokens = 150
        else:
            raise ValueError(f"Invalid target: {target_name}")

        if "api" in target_name:
            NUM_INITIAL_PROMPTS = 10
        
        # Initialize external judge (separate from APEX's internal judge)
        external_judge = Judge(devices=judge_devices)
        
        # Initialize APEX method (self-contained with internal judge)
        apex_method = APEX(max_new_tokens=max_new_tokens, devices=apex_devices)
        
        # Initialize orchestrator with all components
        attacker = ApexRunner(target, external_judge, apex_method, num_sample)

        for category_idx, category_name in enumerate(category_names):
            initial_prompts = content_manager.get_initial_prompts(category_name, NUM_INITIAL_PROMPTS)
            output_folder = os.path.join(base_folder, target_name, category_name)
            os.makedirs(output_folder, exist_ok=True)

            for i, prompt in enumerate(initial_prompts):
                try:
                    output_subfolder = os.path.join(output_folder, f"prompt_{i+1}")
                    
                    if os.path.exists(output_subfolder) and os.path.exists(os.path.join(output_subfolder, "results.json")):
                        with open(os.path.join(output_subfolder, "results.json"), "r") as f:
                            results = json.load(f)
                        if len(results) >= num_round:
                            print(f"  Skipping prompt {i+1} - already completed {len(results)} rounds")
                            continue
                        
                    os.makedirs(output_subfolder, exist_ok=True)
                
                    attacker.attack_by_category(prompt, output_subfolder, category_name, num_round)
               
                except Exception as e:
                    print(f"Error processing prompt {i+1}: {e}")
                    continue
            
            print(f"Completed category: {category_name}")
        
        print("Red-team framework execution completed successfully!")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise
                
    finally:
        # Ensure cleanup happens even if there's an error
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    main()