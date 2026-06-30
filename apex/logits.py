from transformers import LogitsProcessor
import torch
from .utils import *
import spacy
import json
from utils.utils import flatten_list



class AdaptiveLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor that suppresses specified token IDs from being generated.

    Args:
        suppressed_tokens (list or set of int): The token IDs to suppress.
        suppress_value (float, optional): The value to assign to suppressed token logits.
            Defaults to -float("inf") to effectively block generation.
    """

    def __init__(self, processor, alpha: float = 1.0, beta: float = 0.5):
        self.processor = processor
        self.alpha = alpha
        self.beta = beta

        self.counter = {}

        self.nlp = spacy.load("en_core_web_sm")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Modifies the logits by setting the scores of the suppressed tokens to the suppress_value.

        Args:
            input_ids (torch.LongTensor): Generated token IDs so far (shape: [batch_size, sequence_length]).
            scores (torch.FloatTensor): Logits for the next token (shape: [batch_size, vocab_size]).

        Returns:
            torch.FloatTensor: The modified logits with suppressed tokens.
        """
        # Calculate the mean logit value of the input_ids
        input_logits = scores.gather(1, input_ids)
        mean_logit = input_logits.mean().item()
        logit_std = input_logits.std().item()

        # Loop over each token to suppress and set its logit to suppress_value.
        for token_id, token_scores in self.counter.items():
            
            mean_score = sum(token_scores) / len(token_scores) if len(token_scores) > 0 else 0
            count = len(token_scores)
            
            repetition_penalty = 1 / (1 + self.beta * count) if count > 0 and mean_score > 0 else 1.0
            value = mean_score * self.alpha * logit_std * repetition_penalty
            scores[:, token_id] += value
            
        return scores

    def extract_tokens(self, prompt: str):
        trimmed_prompt = self.remove_meaningless_words(prompt)
        tokens = flatten_list(
            flatten_list(
                [
                    self.processor(text=trimmed_prompt, add_special_tokens=False)[
                        "input_ids"
                    ]
                ]
            )
        )
        return tokens
    
    def update(self, prompt: str, score: float):

        tokens = self.extract_tokens(prompt)
        for token_id in tokens:
            if token_id not in self.counter:
                self.counter[token_id] = []
            self.counter[token_id].append(score)
        
            
    def reset(self):
        self.counter.clear()
        
    def save_to(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.counter, f)
            

    def remove_meaningless_words(self, text: str):
        doc = self.nlp(text.lower())
        filtered_words = [
            token.text for token in doc if not token.is_stop and not token.is_punct
        ]
        return " ".join(filtered_words)


