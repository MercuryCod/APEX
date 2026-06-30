from .target import Target
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class SDAPI(Target):
    def __init__(self):
        
        self.api_key_sd = os.getenv("STABLE_DIFFUSION_API_KEY")
        if not self.api_key_sd:
            print("WARNING: STABLE_DIFFUSION_API_KEY not set. API calls will fail.")
        print("Stable Diffusion API initialized")

    def generate_image(self, prompt: str, output_path: str) -> bool:
        try:
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": f"Bearer {self.api_key_sd}",
                    "accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": prompt,
                    "output_format": "png",
                    "model": "sd3.5-large",
                },
            )

            # Handle successful response
            if response.status_code == 200:
                # Save the image
                with open(output_path, "wb") as file:
                    file.write(response.content)
                return True
            
            elif response.status_code == 403:
                # If the response indicates failure
                print(f"Image flagged: {response.status_code}, {response.json()}")
                return False
            else:
                print(f"An error occurred: {response.status_code}, {response.json()}")
                return False

        except Exception as e:
            # Handle any exceptions during the process
            print(f"An error occurred: {e}")
            return False