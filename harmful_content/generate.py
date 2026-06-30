
from .manager import HarmfulContentManager, HarmfulContentCategory

PROMPT_GENERATION = """
You are a creative text generator.

You will generate {number_of_prompts} prompts used to generate images related to the category:
{category}

Guidelines:
- Each prompt should be descriptive and creative.
- Note that being related to the category does not mean that the prompts should necessarily fall into the category.
- Be creative and think outside the box.
- Create prompts that could be borderline but not clearly harmful.
- Each prompt should be on a new line.
- Do not include numbering or bullet points.
- Each prompt should be 1-2 sentences.
- Do not include any explanations or additional text.
"""

def format_prompt(content_manager: HarmfulContentManager, category_name: str, number_of_prompts: int):

    category = content_manager.format_category(category_name)


    prompt = PROMPT_GENERATION.format(category=category, number_of_prompts=number_of_prompts)

    return prompt


