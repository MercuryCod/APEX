"""
Configurable APEX implementation for ablation studies.
Target model is fixed to safe-sd-v2-1 for all experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apex import APEX as BaseAPEX
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    Gemma3ForConditionalGeneration
)
import torch

class ConfigurableAPEX(BaseAPEX):
    """
    Extended APEX class with configurable parameters for ablation studies.
    This class inherits from the original APEX implementation and allows for swapping
    models and adjusting hyperparameters without modifying the original code.
    
    Target model is fixed to safe-sd-v2-1 for all ablation experiments.
    """
    
    def __init__(self, max_new_tokens: int, devices: list[str], 
                 alpha: float = 0.5, beta: float = 0.5, 
                 image_weight: float = 0.8,
                 policy_model: str = "llava-1.6-mistral-7b",
                 judge_model: str = "gemma-3-4b"):
        """
        Initialize configurable APEX with ablation parameters.
        
        Args:
            max_new_tokens: Maximum tokens for generation
            devices: List of CUDA devices [policy_device, judge_device]
            alpha: Logits processor learning rate.
            beta: Logits processor decay factor.  
            image_weight: Weight for image scores (text_weight = 1 - image_weight).
            policy_model: Options: "llava-1.6-mistral-7b", "qwen2.5-vl-7b", "gemma-3-4b", "gemma-3-12b".
            judge_model: Options: "gemma-3-4b", "gemma-3-12b", "qwen2.5-vl-7b".
        """
        print(f"Initializing ConfigurableAPEX with alpha={alpha}, beta={beta}, image_weight={image_weight}")
        print(f"Policy model: {policy_model}, Judge model: {judge_model}")
        print(f"Target: safe-sd-v2-1 (fixed for all ablation studies)")
        print("🔒 All other settings (prompts, tokens, devices) identical to baseline for valid ablation")
        
        # Initialize base APEX. This will load the default models.
        super().__init__(max_new_tokens, devices)
        
        # Override configurable parameters.
        self.logits_processor.alpha = alpha
        self.logits_processor.beta = beta
        self.image_score_weight = image_weight
        self.prompt_score_weight = 1.0 - image_weight
        
        # Store configuration for logging
        self.config = {
            "alpha": alpha,
            "beta": beta,
            "image_weight": image_weight,
            "policy_model": policy_model,
            "judge_model": judge_model,
            "target": "safe-sd-v2-1"
        }
        
        # Load alternative models if specified, replacing the defaults.
        if policy_model != "llava-1.6-mistral-7b":
            self._load_alternative_policy(policy_model)
        if judge_model != "gemma-3-4b":
            self._load_alternative_judge(judge_model)
    
    def _load_alternative_policy(self, model_name: str):
        """Load an alternative policy generation model, releasing memory first."""
        # Clean up existing model to free VRAM
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()

        model_id_map = {
            "gemma-3-4b": "google/gemma-3-4b-it",
            "gemma-3-12b": "google/gemma-3-12b-it",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
        model_id = model_id_map.get(model_name)
        if not model_id:
            raise ValueError(f"Unknown policy model: {model_name}")

        print(f"Loading alternative policy model: {model_id}")
        if "qwen" in model_id.lower():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map=self.devices[0]
            )
        else: # Gemma models
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map=self.devices[0]
            ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def _load_alternative_judge(self, model_name: str):
        """Load an alternative internal judge model, releasing memory first."""
        # Clean up existing agent to free VRAM
        if hasattr(self, 'judge_agent'):
            del self.judge_agent
            torch.cuda.empty_cache()

        # Import agents from the dedicated ablation models directory
        # Note: This requires implementing these classes in the specified file.
        from .models.alternative_judges import QwenJudgeAgent, GemmaJudgeAgent

        print(f"Loading alternative judge model: {model_name}")
        if model_name == "qwen2.5-vl-7b":
            self.judge_agent = QwenJudgeAgent(device=self.devices[1])
            
        elif model_name == "gemma-3-12b":
            model_id = "google/gemma-3-12b-it"
            self.judge_agent = GemmaJudgeAgent(model_id=model_id, device=self.devices[1])
        else:
            raise ValueError(f"Unknown judge model: {model_name}")
            
    def get_config(self):
        """Return current configuration for logging and analysis."""
        return self.config.copy() 