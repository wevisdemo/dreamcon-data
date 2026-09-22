import os
from pathlib import Path
import pandas as pd
from constants import TOPIC_DATA_URL, TOPIC_DATA_COLUMNS


def load_topic_data() -> pd.DataFrame:
    # Load the data directly into a dataframe
    df = pd.read_csv(TOPIC_DATA_URL)
    
    assert all(col in df.columns for col in TOPIC_DATA_COLUMNS), \
    f"Missing columns : {', '.join(set(TOPIC_DATA_COLUMNS) - set(df.columns))}"
    
    # Process categories
    df["category"] = df["categories"].str.split(r"\s*,\s*")
    df = df.explode("category", ignore_index=True)
    
    # Drop categories
    df.drop(columns=['categories'], inplace=True)

    return df