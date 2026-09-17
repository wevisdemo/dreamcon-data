import os, time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import pandas as pd
from io import StringIO

import functools # Important for preserving function metadata (like name, docstring)
def ensure_min_duration(min_duration_seconds):
    """
    Decorator factory that returns a decorator.
    The decorator ensures the decorated function takes at least min_duration_seconds.
    """
    def decorator(func):
        @functools.wraps(func) # Preserves the original function's name, docstring, etc.
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Call the original function
            result = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time

            time_to_sleep = min_duration_seconds - elapsed_time

            if time_to_sleep > 0:
                print(f"'{func.__name__}' finished in {elapsed_time:.4f}s (< {min_duration_seconds}s). Sleeping for {time_to_sleep:.4f}s.")
                time.sleep(time_to_sleep)
            else:
                print(f"'{func.__name__}' finished in {elapsed_time:.4f}s (>= {min_duration_seconds}s). No sleep needed.")

            return result # Return whatever the original function returned

        return wrapper # The decorator returns the new wrapper function
    return decorator # The factory returns the decorator

MODEL_NAME = "gemini-3-flash-preview"

class ModelManager():
    
    _embedding_model = None
    _gg_ai_client = None
    
    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            load_dotenv()
            HF_TOKEN = os.getenv('HF_TOKEN')
            print("Initiate new embedding model...")
            cls._embedding_model = SentenceTransformer(
                'kornwtp/ConGen-model-wangchanberta',
                token=HF_TOKEN
            )
        return cls._embedding_model
    
    @classmethod
    def get_google_ai_client(cls) -> genai.Client:
        if cls._gg_ai_client is None:
            load_dotenv()
            API_KEY = os.getenv("GOOGLE_API_KEY")
            # load GenAI client
            cls._gg_ai_client = genai.Client(api_key=API_KEY)
        return cls._gg_ai_client
    
    @classmethod
    @ensure_min_duration(6)
    def extract_text_from_file(
        cls,
        file_path,
        prompt,
        temperature=0.1
    ):
        print(f"Uploading file : {file_path} ...")
        # Initiate client
        client = cls.get_google_ai_client()
        # upload a file
        sample_pdf = client.files.upload(
            file=file_path, 
            # mime_type="application/pdf",
        )
        
        # 2. Wait for the file to be processed
        # This is critical for PDFs!
        while sample_pdf.state.name == "PROCESSING": # type: ignore
            print(".", end="", flush=True)
            time.sleep(2)
            sample_pdf = client.files.get(name=sample_pdf.name) # type: ignore
        print(f"Upload completed!!")

        # try:
        print(f"Process with {MODEL_NAME} ...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                sample_pdf,
                prompt
            ],
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                temperature=temperature,
        )
        )
        
        return pd.read_csv(StringIO(response.text))
        
        
                