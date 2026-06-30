import os
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*lora_A.default.weight.*")

from Judge import Judge
from Target import Target

from ..base_attacker import BaselineAttacker
from .utils import (
    build_llama_writer,
    build_llava_guide,
    infererence_llama,
    infererence_llava,
)


# Model paths used by ART (LoRA adapters published with the original ART paper).
LLAVA_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_LORA_PATH = "glli/ART_GuideModel"
LLAMA_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LLAMA_LORA_PATH = "glli/ART_WriterModel"


class ART(BaselineAttacker):
    """
    ART baseline: a LoRA-finetuned Llama "writer" rewrites the prompt based on
    instructions produced by a LoRA-finetuned LLaVA "guide" inspecting the
    most recently generated image. No score-based feedback.
    """

    def __init__(
        self,
        judge: Judge,
        target: Target,
        num_sample: int,
        max_new_tokens: int,
        devices: list[str],
    ):
        super().__init__(judge, target, num_sample)

        if devices is None or len(devices) < 2:
            raise ValueError("ART requires at least 2 devices (Llama + LLaVA)")

        self.llama_model, self.llama_tokenizer = build_llama_writer(
            LLAMA_MODEL_PATH, LLAMA_LORA_PATH
        )
        self.llama_model.to(devices[0])

        self.llava_model, self.llava_processor = build_llava_guide(
            LLAVA_MODEL_PATH, LLAVA_LORA_PATH
        )
        self.llava_model.to(devices[1])

        self.max_new_tokens = max_new_tokens

    def attack_by_category(
        self,
        initial_prompt: str,
        output_folder: str,
        category_name: str,
        max_round: int,
    ):
        category = self.content_manager.get_category(category_name)

        os.makedirs(output_folder, exist_ok=True)

        logs = []

        curr_prompt = infererence_llama(
            self.llama_model, self.llama_tokenizer, initial_prompt, category
        )

        logs.append(
            {
                "num_round": 0,
                "prompt": curr_prompt,
                "image_path": None,
                "report": None,
            }
        )

        for num_round in range(1, max_round + 1):
            curr_image_path, curr_report = self.sample_images(
                curr_prompt, output_folder, num_round, category_name
            )

            if curr_image_path is None:
                # Image was filtered or failed; try a fresh rewrite without LLaVA feedback.
                curr_prompt = infererence_llama(
                    self.llama_model, self.llama_tokenizer, curr_prompt, category
                )
            else:
                # Original ART loop: feed the LLaVA instruction string in as the
                # `prompt` arg of infererence_llama (no separate llava_instruction).
                # Matches the published red-team behavior — keep as-is.
                llava_return = infererence_llava(
                    self.llava_model,
                    self.llava_processor,
                    curr_prompt,
                    curr_image_path,
                    category,
                )
                curr_prompt = infererence_llama(
                    self.llama_model,
                    self.llama_tokenizer,
                    llava_return,
                    category,
                )

            logs.append(
                {
                    "num_round": num_round,
                    "prompt": curr_prompt,
                    "image_path": curr_image_path,
                    "report": curr_report,
                }
            )

            print(f"Prompt {num_round}: {curr_prompt}")
            self.print_report(curr_report)
            print("-" * 100 + "\n")

            if num_round % 5 == 0 or num_round == max_round:
                with open(os.path.join(output_folder, "results.json"), "w") as f:
                    json.dump(logs, f, default=str)

        return logs
