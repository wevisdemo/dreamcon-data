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
        
    # Modify propmt to ensure categories
    PROMPT += str(", ".join(
        [f"`{_}`" for _ in df['category'].unique()]
    )).strip(", ")
    
    expected_total_rows = df.groupby(['category', 'group']).size()
    
    PROMPT += f"\nTotal rows of the result have to be : {expected_total_rows}"
    
    print(PROMPT)    
    
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