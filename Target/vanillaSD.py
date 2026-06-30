from .target import Target
from diffusers import StableDiffusion3Pipeline
import torch


class VanillaSD(Target):
    def __init__(self, device: str="balanced"):
        model_path = "stabilityai/stable-diffusion-3.5-large"
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.num_inference_steps = 28
            
            
            
    def generate_image(self, prompt: str, output_path: str) -> bool:
        try:
            
            image = self.pipe(prompt, num_inference_steps=self.num_inference_steps).images[0]
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False