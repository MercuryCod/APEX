from .target import Target
from diffusers import DiffusionPipeline
import torch

class SafeSD(Target):
    def __init__(self, version: str, device: str="balanced"):
       
        if version == "v1-5":
            model_path = "Visualignment/safe-stable-diffusion-v1-5"
        elif version == "v2-1":
            model_path = "Visualignment/safe-stable-diffusion-v2-1"
        else:
            raise ValueError(f"Invalid version: {version}")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        
        print(f"SafeSD {version} initialized")
        
    def generate_image(self, prompt: str, output_path: str) -> bool:
        try:
            image = self.pipe(prompt).images[0]
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False