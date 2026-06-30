from .target import Target
import torch
from diffusers import FluxPipeline

class VanillaFlux(Target):
    def __init__(self, device: str="balanced"):
        model_path = "black-forest-labs/FLUX.1-schnell"
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device)
        self.pipe.enable_model_cpu_offload()
        
        print("FLUX.1-schnell initialized")

    def generate_image(self, prompt: str, output_path: str) -> bool:
        try:
            out = self.pipe(
                prompt=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
            ).images[0]
            out.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False