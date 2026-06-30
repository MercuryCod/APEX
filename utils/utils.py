
import re
import json5
import dirtyjson
import ast
from PIL import Image
import base64
import json
from typing import Any, Optional



def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def normalize_quotes(text):
    """
    Replace all curly quotes (single and double) with standard ASCII straight quotes.
    """
    replacements = {
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '„': '"',  # double low-9 quotation mark (German)
    }

    for curly, straight in replacements.items():
        text = text.replace(curly, straight)
    
    return text
    
def safe_parse_json(json_string: str) -> Optional[Any]:
    """
    Robustly convert string to JSON with multiple fallback methods.

    Args:
        json_string: String potentially containing JSON data

    Returns:
        Parsed JSON object or None if parsing fails
    """
    # Remove Markdown code block formatting
    json_string = json_string.strip()
    json_string = re.sub(r"^```(?:json)?", "", json_string, flags=re.IGNORECASE).strip()
    json_string = re.sub(r"```$", "", json_string).strip()

    json_string = normalize_quotes(json_string)
    
    # Try json
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    
    # Try json in markdown code block
    try:
        json_match = re.search(r'```json\n(.*?)\n```', json_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
    except Exception:
        pass

    # Try json5 (more lenient)
    try:
        return json5.loads(json_string)
    except Exception:
        pass

    # Try literal_eval
    try:
        return ast.literal_eval(json_string)
    except Exception:
        pass

    
    try:
        return dirtyjson.loads(json_string)
    except Exception:
        pass

    return None


def select_best_image(image_reports: list[tuple[str, dict]]) -> tuple[str, dict]:
    """
    Select the best image from a list of (image_path, report) tuples.
    Prefers unsafe images; among same safety label, selects highest score.

    Args:
        image_reports: List of (image_path, report_dict) tuples.

    Returns:
        (best_image_path, best_report) or (None, None) if no valid reports.
    """
    best_path = None
    best_report = None
    max_score = float("-inf")

    for image_path, report in image_reports:
        if report is None:
            continue

        score = report.get("score", -1)
        label = report.get("label", "safe")

        is_current_unsafe = (label == "unsafe")
        is_best_unsafe = (best_report is not None and best_report.get("label") == "unsafe")

        should_update = False
        if is_current_unsafe and not is_best_unsafe:
            should_update = True
        elif is_current_unsafe and is_best_unsafe:
            should_update = (score > max_score)
        elif not is_current_unsafe and not is_best_unsafe:
            should_update = (score > max_score)

        if should_update:
            max_score = score
            best_path = image_path
            best_report = report

    return best_path, best_report
