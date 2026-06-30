"""
Entry point for running baseline red-teaming methods (ART, Groot, Flirt).

APEX itself is run via `main.py`. This script exists to give collaborators a
clean way to compare APEX against the baselines on the same targets, judges,
and harmful-content categories.
"""
import argparse
import json
import os
import warnings

import torch

from Judge import Judge
from Target import SDAPI, SafeSD, VanillaSD, VanillaFlux
from apex.judge_agent import Judge_Agent
from baselines import ART, Groot, Flirt
from harmful_content import HarmfulContentManager

warnings.filterwarnings("ignore")


def build_target(target_name: str):
    if target_name == "sd-3.5-large":
        return VanillaSD(), 77
    if target_name == "safe-sd-v2-1":
        return SafeSD("v2-1"), 77
    if target_name == "safe-sd-v1-5":
        return SafeSD("v1-5"), 77
    if target_name == "flux":
        return VanillaFlux(), 77
    if target_name == "sd-api":
        return SDAPI(), 150
    raise ValueError(f"Invalid target: {target_name}")


def build_attacker(method: str, judge: Judge, target, num_sample: int, max_new_tokens: int, method_devices: list[str]):
    if method == "art":
        if len(method_devices) < 2:
            raise ValueError("ART requires 2 method devices (Llama writer + LLaVA guide)")
        return ART(judge, target, num_sample, max_new_tokens, devices=method_devices[:2])

    if method == "groot":
        return Groot(judge, target, num_sample, max_new_tokens, device=method_devices[0])

    if method == "flirt":
        if len(method_devices) < 2:
            raise ValueError("Flirt requires 2 method devices (LLaVA + Judge_Agent)")
        judge_agent = Judge_Agent(method_devices[1])
        return Flirt(
            judge,
            judge_agent,
            target,
            num_sample,
            max_new_tokens,
            device=method_devices[0],
        )

    raise ValueError(f"Invalid method: {method}")


def main():
    parser = argparse.ArgumentParser(description="APEX baseline red-team runner")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["art", "groot", "flirt"],
        help="Baseline method to run",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        required=True,
        choices=["sd-3.5-large", "safe-sd-v2-1", "safe-sd-v1-5", "sd-api", "flux"],
        help="Target text-to-image model",
    )
    parser.add_argument("--num_round", type=int, default=20, help="Attack rounds per prompt")
    parser.add_argument("--num_sample", type=int, default=2, help="Images sampled per round")
    parser.add_argument("--base_folder", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--judge_devices",
        type=str,
        nargs=2,
        default=["cuda:0", "cuda:1"],
        help="Two CUDA devices for the external Judge",
    )
    parser.add_argument(
        "--method_devices",
        type=str,
        nargs="+",
        default=["cuda:2", "cuda:3"],
        help=(
            "CUDA devices for the baseline method. "
            "ART needs 2 (Llama writer, LLaVA guide); "
            "Groot needs 1; Flirt needs 2 (LLaVA, Judge_Agent)."
        ),
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # API targets give fewer initial prompts to keep API costs bounded — same
    # convention as APEX's main.py.
    num_initial_prompts = 10 if "api" in args.target_name else 30

    content_manager = HarmfulContentManager()
    category_names = content_manager.get_all_category_names()

    try:
        target, max_new_tokens = build_target(args.target_name)
        judge = Judge(devices=args.judge_devices)
        attacker = build_attacker(
            args.method, judge, target, args.num_sample, max_new_tokens, args.method_devices
        )

        for category_name in category_names:
            initial_prompts = content_manager.get_initial_prompts(
                category_name, num_initial_prompts
            )
            output_folder = os.path.join(
                args.base_folder, args.target_name, args.method, category_name
            )
            os.makedirs(output_folder, exist_ok=True)

            for i, prompt in enumerate(initial_prompts):
                output_subfolder = os.path.join(output_folder, f"prompt_{i + 1}")

                results_path = os.path.join(output_subfolder, "results.json")
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        results = json.load(f)
                    if len(results) >= args.num_round:
                        print(
                            f"Skipping prompt {i + 1} for {category_name} "
                            f"(already completed {len(results)} rounds)"
                        )
                        continue

                os.makedirs(output_subfolder, exist_ok=True)

                try:
                    attacker.attack_by_category(
                        prompt, output_subfolder, category_name, args.num_round
                    )
                except Exception as e:
                    print(f"Error processing prompt {i + 1} for {category_name}: {e}")
                    continue

            print(f"Completed category: {category_name}")

        print(f"Baseline {args.method} run complete.")

    finally:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
