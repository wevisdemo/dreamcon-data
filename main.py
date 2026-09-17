import os
import pandas as pd
from pathlib import Path
from agglo_cluster import cluster_topic
from question_generator import generate_question

data_dir_path = Path(__file__).resolve().parent / "data"
output_dir_path = Path(__file__).resolve().parent / "output"

def main() -> None:
    print("Hello DreamCon")
    
    # Check & Create output directory
    os.makedirs(data_dir_path, exist_ok=True)
    
    # Load topic data
    topic_df = pd.read_csv(data_dir_path / "topics.csv")
    
    clustered_topic_df = cluster_topic(topic_df)
    
    full_questions_df = generate_question(clustered_topic_df)
    
    # Save to csv
    clustered_topic_df.to_csv(output_dir_path / "topic_groups.csv", index=False)
    full_questions_df.to_csv(output_dir_path / "group_questions.csv", index=False)
    
if __name__ == "__main__":
    main()
