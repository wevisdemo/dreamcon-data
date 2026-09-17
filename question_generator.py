import os, time
import tempfile
import pandas as pd
from model_manager import ModelManager

def generate_question(df: pd.DataFrame, prompt_path:str|None=None) -> pd.DataFrame:
    
    if prompt_path:
        with open(prompt_path, "r") as file:
            PROMPT = file.read()
    elif os.path.exists("data/prompt/prompt.txt"):
        with open("data/prompt/prompt.txt", "r") as file:
            PROMPT = file.read()
    else:
        PROMPT = ""
    
    # Create temp .txt file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w+t', delete=True) as tf:
        
        df.to_csv(tf, index=False)
        
        print(f"TemporaryFile CSV created at: {tf.name}")
        
        result = ModelManager.extract_text_from_file(
            file_path=tf.name,
            prompt=PROMPT
        )
        
        tf.close()
    
    return result