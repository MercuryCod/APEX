from abc import ABC, abstractmethod
import os


class Target(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_image(self, prompt: str, output_path: str) -> bool:
        pass
    
    def sample_images(self, prompt: str, output_folder_path: str, num_round: int, num_sample: int) -> list[str]:
        num_sample = max(num_sample, 1)
        output_folder_path = os.path.join(output_folder_path, f"round_{num_round}")
        os.makedirs(output_folder_path, exist_ok=True)
        
        image_paths = []
        for i in range(num_sample):
            image_path = os.path.join(output_folder_path, f"image_{i}.png")
            if not self.generate_image(prompt, image_path):
                print(f"Error generating image: {image_path}")
            else:
                image_paths.append(image_path)

        return image_paths
