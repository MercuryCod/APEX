from PIL import Image
import os
import logging
import torch
from utils.image_utils import load_image

logger = logging.getLogger(__name__)

def postprocess_llava_7b(response: str):
    logger.debug("Raw LLaVA response (%d chars): %s", len(response), response)

    # Look for the actual generated content after the template
    if "[/INST]" in response:
        result = response.split("[/INST]")[1].strip().strip('"')
        logger.debug("After [/INST] split: %s", result)
    elif "Generate only the new image prompt, nothing else:" in response:
        # Try to extract content after our instruction
        parts = response.split("Generate only the new image prompt, nothing else:")
        if len(parts) > 1:
            result = parts[1].strip().strip('"')
            logger.debug("After instruction split: %s", result)
        else:
            logger.debug("Found instruction but no content after it")
            result = response.strip().strip('"')
    else:
        logger.debug("No split markers found - using full response")
        result = response.strip().strip('"')

    return result

def generate_output_with_logits_llava(
    model,
    processor,
    prompt: str,
    image_path,
    logits_processor,
    max_new_tokens: int = 150,
):
    logger.debug("LLaVA Generation Input: prompt_len=%d, image=%s, max_tokens=%d",
                 len(prompt), image_path, max_new_tokens)
    logger.debug("Policy prompt preview: %.500s...", prompt)

    if image_path is not None and os.path.exists(image_path):
        image = Image.open(image_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    context_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    if image_path is not None and os.path.exists(image_path):
        inputs = processor(
            images=[image], text=context_prompt, return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=context_prompt, return_tensors="pt"
        ).to(model.device)

    logger.debug("Chat template applied (%d chars), preview: ...%s",
                 len(context_prompt), context_prompt[-200:])

    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=max_new_tokens,
        logits_processor=[logits_processor],
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    output_str = processor.decode(output[0], skip_special_tokens=True)

    logger.debug("Model output (%d chars), preview: ...%s",
                 len(output_str), output_str[-300:])

    result = postprocess_llava_7b(output_str)

    logger.debug("Final extracted prompt (%d chars): %s", len(result), result)

    return result



def generate_output_gemma(model, processor, prompt, image_path, max_new_tokens=600):
    if image_path is None or not os.path.exists(image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
    else:
        image = load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="longest",
        pad_to_multiple_of=8,
    ).to(model.device, dtype=torch.bfloat16)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    decoded = processor.decode(output[0], skip_special_tokens=True)

    parts = decoded.split("model\n")
    if len(parts) > 1:
        output_str = parts[1].strip().strip("\"'")
    else:
        output_str = decoded.strip().strip("\"'")

    return output_str
