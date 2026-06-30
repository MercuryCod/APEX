from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.model_utils import generate_output_with_logits_llava
import torch
import os
from .logits import AdaptiveLogitsProcessor
from .judge_agent import Judge_Agent
from .utils import get_policy_prompt
from utils.utils import select_best_image
from harmful_content import HarmfulContentManager


def get_prompt_image_scores(prompt_report, image_report):
    """Extract scores from prompt and image analysis reports."""
    prompt_score = prompt_report.get("score", 0) if prompt_report else 0
    
    if image_report is None:
        image_score = -1
        return prompt_score, image_score
    
    image_score = image_report.get("score", 0)
    return prompt_score, image_score


class APEX:
    """
    Complete APEX (Adaptive Prompt Evolution with eXpert guidance) implementation.
    
    This class handles all APEX-specific logic:
    - LLaVA-based prompt generation with logits processing
    - Internal judge agent for analyzing prompts and images
    - Scoring and feedback integration
    - Logits processor state management
    """
    
    def __init__(self, max_new_tokens: int, devices: list[str]):
        """
        Initialize the APEX method with all required components.
        
        Args:
            max_new_tokens: Maximum tokens for prompt generation
            devices: List of CUDA devices for model loading
        """
        model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        self.max_new_tokens = max_new_tokens
        self.num_past_prompts = 3
        self.devices = devices
        
        # Initialize LLaVA model for prompt generation
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            do_image_splitting=False,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.devices[0],
            use_safetensors=True,
        )
        
        # Initialize logits processor for adaptive prompt generation
        self.logits_processor = AdaptiveLogitsProcessor(self.processor, alpha=0.5, beta=0.5)
        
        # Initialize internal judge agent (separate from external judge)
        self.judge_agent = Judge_Agent(self.devices[1])
        
        # Initialize content manager for category handling
        self.content_manager = HarmfulContentManager()
        
        # Scoring weights for combining prompt and image scores
        self.image_score_weight = 0.8
        self.prompt_score_weight = 0.2
    
    def generate_prompt(self, policy_prompt: str, image_path: str) -> str:
        """
        Generate a new adversarial prompt using LLaVA with logits processing.
        
        Args:
            policy_prompt: The policy-guided prompt template
            image_path: Path to the current image for context
            
        Returns:
            Generated adversarial prompt string
        """
        return generate_output_with_logits_llava(
            self.model, 
            self.processor, 
            policy_prompt, 
            image_path, 
            self.logits_processor, 
            self.max_new_tokens
        )
    
    def analyze(self, prompt: str, image_paths: list, category_name: str) -> dict:
        """
        Analyze prompt and images using APEX's internal judge agent.
        
        Args:
            prompt: The text prompt to analyze
            image_paths: List of image paths to analyze
            category_name: Harmful content category for analysis
            
        Returns:
            Dictionary with prompt analysis and best image analysis
        """
        # Analyze the prompt using internal judge
        prompt_report = self.judge_agent.analyze_text(prompt, self.content_manager, category_name)
        
        # Analyze images and find the best one based on score
        image_reports = []
        for image_path in image_paths:
            if image_path is None or not os.path.exists(image_path):
                continue
            image_report = self.judge_agent.analyze_image(image_path, self.content_manager, category_name)
            image_reports.append((image_path, image_report))

        best_image_path, best_image_report = select_best_image(image_reports)
        
        return {
            "prompt_report": prompt_report,
            "image_report": best_image_report,
            "best_image_path": best_image_path
        }
    
    def calculate_score(self, prompt_report: dict, image_report: dict) -> float:
        """
        Calculate combined score from prompt and image analysis reports.
        
        Args:
            prompt_report: Analysis report for the prompt
            image_report: Analysis report for the image  
            image_path: Path to the image (None if no image generated)
            
        Returns:
            Combined score for feedback to logits processor
        """
        prompt_score, image_score = get_prompt_image_scores(prompt_report, image_report)
            
        # Weighted combination of prompt and image scores
        score = self.image_score_weight * image_score + self.prompt_score_weight * prompt_score
        return score
    
    def update_logits(self, prompt: str, score: float):
        """
        Update the logits processor with prompt and score feedback.
        
        Args:
            prompt: The prompt that was used
            score: The calculated score for this prompt
        """
        self.logits_processor.update(prompt, score)
    
    def reset(self):
        """Reset the logits processor for a new attack sequence."""
        self.logits_processor.reset()
    
    def save_logits(self, path: str):
        """
        Save the current logits processor state.
        
        Args:
            path: File path to save logits data
        """
        self.logits_processor.save_to(path)
    
    def build_policy_prompt(self, logs: list, category_str: str) -> str:
        """
        Build the policy prompt for the next iteration using attack history.
        
        Args:
            logs: List of previous attack results
            category_str: Formatted category information
            
        Returns:
            Policy prompt string for next iteration
        """
        return get_policy_prompt(logs, category_str, self.num_past_prompts)
