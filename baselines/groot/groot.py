import os
import random

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from Judge import Judge
from Target import Target
from utils.utils import safe_parse_json

from ..base_attacker import BaselineAttacker
from .utils import (
    decompose_prompt,
    decompose_leaf_node,
    get_leaf_nodes,
    count_objects,
    reconstruct_prompt,
)


LLAVA_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"


class Groot(BaselineAttacker):
    """
    Groot baseline: decomposes the prompt into a JSON "Prompt Parse Tree",
    randomly selects a leaf to be re-described in non-sensitive terms, then
    reconstructs the prompt. Open-loop — image is generated for evaluation
    but never feeds back into the next prompt.
    """

    def __init__(
        self,
        judge: Judge,
        target: Target,
        num_sample: int,
        max_new_tokens: int,
        device: str = None,
    ):
        super().__init__(judge, target, num_sample)

        self.processor = AutoProcessor.from_pretrained(
            LLAVA_MODEL_PATH,
            do_image_splitting=False,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            LLAVA_MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device if device is not None else "auto",
            use_safetensors=True,
        )

        self.max_new_tokens = max_new_tokens

    def generate_prompt(self, prompt: str) -> str:
        try:
            processed_prompt = decompose_prompt(self.model, self.processor, prompt)
            json_data = safe_parse_json(processed_prompt)

            leaf_nodes = get_leaf_nodes(json_data)
            selected_node = random.choice(leaf_nodes)

            obj_count = count_objects(json_data)
            decomposed_node = decompose_leaf_node(
                self.model, self.processor, selected_node, obj_count
            )

            json_data[selected_node["index"]] = decomposed_node

            reconstructed_prompt = reconstruct_prompt(
                self.model,
                self.processor,
                json_data,
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
            return reconstructed_prompt

        except Exception:
            # If decomposition / reconstruction fails for any reason, fall back
            # to the previous prompt — preserves the original baseline behavior.
            return prompt

    def attack_by_category(
        self,
        initial_prompt: str,
        output_folder: str,
        category_name: str,
        max_round: int,
    ):
        os.makedirs(output_folder, exist_ok=True)
        logs = []

        curr_prompt = initial_prompt

        for num_round in range(1, max_round + 1):
            curr_image_path, curr_report = self.sample_images(
                curr_prompt, output_folder, num_round, category_name
            )

            logs.append(
                {
                    "num_round": num_round,
                    "prompt": curr_prompt,
                    "image_path": curr_image_path,
                    "report": curr_report,
                }
            )

            curr_prompt = self.generate_prompt(curr_prompt)

            print(f"Prompt {num_round}: {curr_prompt}")
            self.print_report(curr_report)
            print("-" * 100 + "\n")

            if num_round % 5 == 0 or num_round == max_round:
                self.save_logs(logs, output_folder)

        return logs
