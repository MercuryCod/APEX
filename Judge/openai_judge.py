from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()


_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    print("WARNING: OPENAI_API_KEY not set. Text moderation will fail.")
openai_client = OpenAI(api_key=_openai_api_key)


def judge_text(text, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            response = openai_client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )
            return response.results[0].flagged
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return False
