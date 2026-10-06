import pandas as pd
from sentence_transformers import util
from model_manager import ModelManager

def get_text_distance(text1: str, text2: str) -> float:
    model = ModelManager.get_embedding_model()
    embeddings = model.encode([text1, text2])
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    return 1.0 - cosine_similarity

def calculate_question_distance(topic_df: pd.DataFrame, question_df: pd.DataFrame) -> pd.DataFrame:
    df = topic_df.merge(
        question_df[['category', 'group', 'question']], 
        on=['category', 'group'], 
        how='left'
    )
    
    df['question'] = df['question'].fillna("")
    
    # Apply the distance function row-by-row (note the axis=1)
    df['embedded_distance_to_question'] = df.apply(
        lambda row: get_text_distance(row['title'], row['question']),
        axis=1
    )
    
    df.drop(columns=['question'])
    
    return df