import os
import base64
from PIL import Image

def is_valid_image_path(image_path: str):
    if image_path is None or not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist")
        return False
    return True

def load_image(image_path: str):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    return image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_image_sd_local(pipe, prompt, output_path) -> bool:
    image = pipe(prompt).images[0]
    image.save(output_path)
    return True