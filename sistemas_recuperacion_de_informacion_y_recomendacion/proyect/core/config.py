import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
