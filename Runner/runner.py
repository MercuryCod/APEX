import os
import json
from apex import APEX
from Judge import Judge
from harmful_content import HarmfulContentManager
from Target import Target
from utils.utils import select_best_image


class ApexRunner:
    """
    Orchestrator for APEX red-teaming attacks.
    """
    
    def __init__(self, target: Target, judge: Judge, apex: APEX, num_sample: int):
        self.judge = judge
        self.target = target
        self.apex = apex
        self.num_sample = num_sample
        self.content_manager = HarmfulContentManager()

    def print_judge_results(self, report: dict, image_name: str):
        """Print external judge results for a single image."""
        if report:
            for judge_name, result in report.items():
                if isinstance(result, dict) and result:
                    label = result.get("label", "unknown")
                    category = result.get("category", "NA")
                    
                    # Only show score if it exists
                    if "score" in result:
                        score = result.get("score")
                        print(f"    {judge_name}: {label} (score: {score:.3f}, category: {category})")
                    else:
                        print(f"    {judge_name}: {label} (category: {category})")
                else:
                    print(f"    {judge_name}: failed")
        else:
            print(f"    No valid evaluation")

    def print_apex_analysis(self, apex_analysis: dict):
        """Print APEX internal analysis results."""
        print("  APEX Internal Analysis:")
        
        # Print prompt analysis
        if apex_analysis.get("prompt_report"):
            prompt_report = apex_analysis["prompt_report"]
            label = prompt_report.get("label", "unknown")
            score = prompt_report.get("score", -1)
            category = prompt_report.get("category", "NA")
            rationale = prompt_report.get("rationale", "No rationale")
            print(f"    Prompt: {label} (score: {score:.3f}, category: {category})")
            print(f"    Rationale: {rationale}")
        
        # Print image analysis
        if apex_analysis.get("image_report"):
            image_report = apex_analysis["image_report"]
            label = image_report.get("label", "unknown")
            score = image_report.get("score", -1)
            category = image_report.get("category", "NA")
            description = image_report.get("description", "No description")
            best_image = apex_analysis.get("best_image_path", "unknown")
            print(f"    Image ({os.path.basename(best_image)}): {label} (score: {score:.3f}, category: {category})")
            print(f"    Description: {description}")
        else:
            print("    Image: No analysis available")

    def sample_images_with_judges(self, prompt: str, output_folder: str, num_round: int, category_name: str):
        """
        Sample images using target model and select best using external judge.
        """
        
        # 1. Generate images
        valid_images = self.target.sample_images(prompt, output_folder, num_round, self.num_sample)
        
        if not valid_images:
            print("No valid images generated")
            return None, None
        
        print(f"{len(valid_images)}/{self.num_sample} images generated")
        print()  # Add spacing
        
        # 3. Evaluate images with external judge
        gemma_reports = []
        full_reports = {}
        for i, image_path in enumerate(valid_images):
            report = self.judge.classify_image(image_path, self.content_manager, category_name)

            print(f"  image_{i}.png External Judge Results:")
            self.print_judge_results(report, f"image_{i}.png")
            print()

            full_reports[image_path] = report
            gemma_report = report.get("gemma", {}) if report else None
            gemma_reports.append((image_path, gemma_report))

        best_image_path, best_gemma_report = select_best_image(gemma_reports)
        best_report = full_reports.get(best_image_path) if best_image_path else None

        if best_image_path is None:
            print("No valid image evaluations from external judge")
            return None, None

        best_score = best_gemma_report.get("score", 0) if best_gemma_report else 0
        print(f"Best image selected: {os.path.basename(best_image_path)} (score: {best_score:.3f})")
        print()  # Add spacing

        return best_image_path, best_report

    def attack_by_category(self, initial_prompt: str, output_folder: str, category_name: str, max_round: int):
        """
        Execute a complete APEX attack for a specific harmful content category.
        """
        # Initialize attack state
        logs = []
        category_str = self.content_manager.format_category(category_name)
        curr_prompt = initial_prompt
        
        self.apex.reset()
        
        # Round 0: Evaluate initial prompt
        num_round = 0
        print("-" * 100)
        print(f"Round {num_round} | Prompt: {curr_prompt}")
        print("-" * 100)
        print()  # Add spacing
        
        # 1. Target generates images, external judge selects best
        curr_image_path, curr_report = self.sample_images_with_judges(
            initial_prompt, output_folder, num_round, category_name
        )
        
        # 2. APEX analyzes prompt and images with internal judge
        apex_analysis = self.apex.analyze(
            curr_prompt, 
            [curr_image_path] if curr_image_path else [], 
            category_name
        )
        
        # Print APEX internal analysis
        self.print_apex_analysis(apex_analysis)
        print()  # Add spacing
        
        # 3. APEX calculates score and updates strategy
        score = self.apex.calculate_score(
            apex_analysis["prompt_report"], 
            apex_analysis["image_report"]
        )
        self.apex.update_logits(curr_prompt, score)
        print(f"APEX feedback score: {score:.3f}")
        print()  # Add spacing
        
        prompt_safety = "unsafe" if self.judge.classify_text(curr_prompt) else "safe"
        
        # Log round results
        result = {
            "round": num_round,
            "prompt": curr_prompt,
            "prompt_safety": prompt_safety,
            "image_path": curr_image_path,
            "judge_report": curr_report,
            "apex_analysis": apex_analysis,
            "apex_score": score
        }
        logs.append(result)
        
        # Rounds 1 to max_round: Iterative attack refinement
        for num_round in range(1, max_round + 1):
            print("-" * 100)
            print(f"Round {num_round} | Generating new adversarial prompt...")
            print("-" * 100)
            print()  # Add spacing

            # 1. APEX builds policy prompt
            policy_prompt = self.apex.build_policy_prompt(logs, category_str)

            # 2. APEX generates new adversarial prompt
            # Use current image if available, otherwise fall back to previous valid image
            image_for_generation = curr_image_path
            if image_for_generation is None and logs:
                # Try to find the most recent valid image from previous rounds
                for prev_log in reversed(logs):
                    prev_image = prev_log.get("image_path")
                    if prev_image and os.path.exists(prev_image):
                        image_for_generation = prev_image
                        print(f"⚠️  Using previous image from round {prev_log.get('round', 'unknown')} as fallback")
                        break

            if image_for_generation is None:
                print(f"⚠️  WARNING: No valid images available for generation in round {num_round}")
                print(f"Skipping to next round...")
                continue

            curr_prompt = self.apex.generate_prompt(policy_prompt, image_for_generation)
            print(f"Generated prompt: {curr_prompt}")
            print()  # Add spacing
            
            # 3. Target generates images, external judge selects best
            curr_image_path, curr_report = self.sample_images_with_judges(
                curr_prompt, output_folder, num_round, category_name
            )
            
            # 4. APEX analyzes new results with internal judge
            apex_analysis = self.apex.analyze(
                curr_prompt,
                [curr_image_path] if curr_image_path else [],
                category_name
            )
            
            # Print APEX internal analysis
            self.print_apex_analysis(apex_analysis)
            print()  # Add spacing
            
            # 5. APEX calculates score and updates strategy
            score = self.apex.calculate_score(
                apex_analysis["prompt_report"],
                apex_analysis["image_report"]
            )
            self.apex.update_logits(curr_prompt, score)
            print(f"APEX feedback score: {score:.3f}")
            
            prompt_safety = "unsafe" if self.judge.classify_text(curr_prompt) else "safe"
            print(f"Prompt Safety: {prompt_safety}")
            print()  # Add spacing
            
            # Log round results
            result = {
                "round": num_round,
                "prompt": curr_prompt,
                "prompt_safety": prompt_safety,
                "image_path": curr_image_path,
                "judge_report": curr_report,
                "apex_analysis": apex_analysis,
                "apex_score": score
            }
            logs.append(result)
        
        # Save final attack summary
        summary = {
            "category": category_name,
            "initial_prompt": initial_prompt,
            "max_rounds": max_round,
            "total_rounds": len(logs),
            "logs": logs
        }
        
        summary_path = os.path.join(output_folder, "attack_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("=" * 100)
        print(f"ATTACK COMPLETE | {len(logs)} rounds | Results saved to {summary_path}")
        print("=" * 100)
        
        return summary
    