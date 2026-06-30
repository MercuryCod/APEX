from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import numpy as np
from utils.image_utils import load_image

class Safety_Filter:
    
    def __init__(self, device: str):
        
        model_path = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = CLIPImageProcessor.from_pretrained(model_path)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            model_path, device_map=device
        )
        self.safety_checker.eval()
        self.labels = ["safe", "unsafe"]

    def classify_image(self, image_path: str):
        image = load_image(image_path)
        safety_checker_input = self.safety_feature_extractor(
            [image], return_tensors="pt"
        ).to(self.safety_checker.device)
        x_checked_image, has_nsfw_concept = self.safety_checker(
            clip_input=safety_checker_input.pixel_values,
            images=np.expand_dims(np.array(image), axis=0),
        )
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        result =self.labels[int(has_nsfw_concept[0])]
        
        return {
            "label": result,
        }
    
