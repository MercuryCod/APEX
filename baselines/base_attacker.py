from abc import ABC, abstractmethod
import os
import json

from harmful_content import HarmfulContentManager
from Judge import Judge
from Target import Target


class BaselineAttacker(ABC):
    """
    Abstract base class for baseline red-teaming attackers (ART, Groot, Flirt).

    Provides shared infrastructure for image sampling, judge scoring, and log
    persistence. Each baseline subclass implements its own `attack_by_category`
    loop, calling `self.sample_images` to drive the target/judge.
    """

    def __init__(self, judge: Judge, target: Target, num_sample: int):
        self.judge = judge
        self.target = target
        self.num_sample = num_sample
        self.content_manager = HarmfulContentManager()

    @abstractmethod
    def attack_by_category(
        self,
        initial_prompt: str,
        output_folder: str,
        category_name: str,
        max_round: int,
    ):
        pass

    def save_logs(self, logs: list, output_folder: str):
        try:
            with open(os.path.join(output_folder, "results.json"), "w") as f:
                json.dump(logs, f, default=str)
        except Exception as e:
            print(f"Error saving logs to {output_folder}: {e}")

    def print_report(self, report: dict):
        if report is None:
            print("No report")
            return

        for key, value in report.items():
            if value is None:
                print(f"{key}: None")
            else:
                print(f"{key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            print()

    def sample_images(
        self,
        prompt: str,
        output_folder: str,
        num_round: int,
        category_name: str,
    ):
        """
        Generate `num_sample` images and select the highest-scoring one
        using the external Judge. Returns (best_image_path, best_report)
        or (None, None) if every generation was filtered/failed.
        """
        image_paths = self.target.sample_images(
            prompt, output_folder, num_round, self.num_sample
        )

        if not image_paths:
            return None, None

        max_score = float("-inf")
        best_image_path = None
        best_report = None

        for image_path in image_paths:
            report = self.judge.classify_image(
                image_path, self.content_manager, category_name
            )
            if report is None:
                continue

            gemma = report.get("gemma") or {}
            score = gemma.get("score", -1)
            label = gemma.get("label", "safe")

            if label == "unsafe" or (score > max_score and label == "safe"):
                max_score = score
                best_image_path = image_path
                best_report = report

        if best_image_path is None:
            return None, None

        return best_image_path, best_report
