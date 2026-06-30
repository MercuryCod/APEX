import torch
from .gemma_judge import Gemma_Judge
from .sd_safety_judge import Safety_Filter
from .llava_guard_judge import Llava_Guard
from .openai_judge import judge_text
from harmful_content import HarmfulContentManager
from utils.image_utils import is_valid_image_path


class Judge:
    def __init__(self, devices: list[str] = ["cuda"]):
        self.gemma_judge = Gemma_Judge(devices[0])
        self.sd_safety_checker = Safety_Filter(devices[0])
        self.llava_guard = Llava_Guard(devices[1])

    def classify_text(self, text: str):
        return judge_text(text)

    def classify_image(self, image_path: str, content_manager: HarmfulContentManager, category_name: str = None):
        gemma_result = None
        llava_guard_result = None

        if is_valid_image_path(image_path):
            if self.gemma_judge:
                try:
                    gemma_result = self.gemma_judge.classify_image(image_path, content_manager, category_name)
                except Exception as e:
                    print(f"Error classifying image with Gemma: {e}")
                    gemma_result = None

            if self.llava_guard:
                try:
                    llava_guard_result = self.llava_guard.classify_image(image_path, content_manager, category_name)
                except Exception as e:
                    print(f"Error classifying image with LLaVA Guard: {e}")
                    llava_guard_result = None

        result = {
            "gemma": gemma_result,
            "llava_guard": llava_guard_result,
        }
        return result
