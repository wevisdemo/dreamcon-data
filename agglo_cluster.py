import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from model_manager import ModelManager

def optimize_agglomerative_clustering_k(
    embeddings, 
    k_values=range(2, 11),  # Default: tests K from 2 to 10
    metric='cosine', 
    linkage='complete'
):
    """
    Iterates through a list of cluster counts (K) to find the optimal 
    Agglomerative Clustering model based on the highest Silhouette Score.
    """
    best_score = -1.0
    best_k = None
    best_labels = None
    best_model = None
    
    # To keep track of all iterations for analysis
    results_history = []

    for k in k_values:
        # Silhouette Score requires at least 2 clusters and at most N-1
        if k < 2 or k >= len(embeddings):
            print(f"Skipping k={k}: Silhouette score requires 2 <= k < n_samples.")
            continue

        clustering_model = AgglomerativeClustering(
            n_clusters=k,              # Set the exact number of clusters you want
            metric=metric,             
            linkage=linkage, # type: ignore
            distance_threshold=None    # Must be None when n_clusters is not None
        )
        
        # Fit the model
        clustering_model.fit(embeddings)
        labels = clustering_model.labels_
        
        # Calculate Silhouette Score using the same metric
        score = silhouette_score(embeddings, labels, metric=metric)
            
        results_history.append({
            'k': k,
            'silhouette_score': score
        })
        
        # Update best model if current score is higher
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = clustering_model

    # If no valid clustering was found
    if best_labels is None:
        print("Could not find a valid clustering setup.")
        return None

    # Analyze best results
    n_noise = np.sum(best_labels == -1)  # Will be 0 for AgglomerativeClustering
    
    print("--- Best Clustering Results ---")
    print(f"Optimal Number of Clusters (K) : {best_k}")
    print(f"Silhouette Score               : {best_score:.4f}")
    print(f"Noise points                   : {n_noise}")
    
    return {
        'model': best_model,
        'labels': best_labels,
        'k': best_k,
        'silhouette_score': best_score,
        'history': results_history
    }
    
def cluster_topic(topics_df: pd.DataFrame) -> pd.DataFrame:
    
    # Initialize new df with the same columns as topics_df
    new_topics_df = pd.DataFrame(columns=topics_df.columns)
    
    # Initialize embedding model
    model = ModelManager.get_embedding_model()

    for category in topics_df['category'].unique():
        
        print(category)
        category_df = topics_df[topics_df['category'] == category]
        
        # Embeddings
        embeddings = model.encode(category_df['title'].to_list())
        
        result = optimize_agglomerative_clustering_k(
            embeddings, 
            k_values=range(2, 10), # Tests 2, 3, 4 ... up to 20 clusters
            metric='cosine',
            linkage='complete'
        )
        
        if not result:
            continue
        
        best_labels = result['labels']
        optimal_k = result['k']
        
        # Add group label
        category_df.loc[category_df.index, ['group']] = best_labels
        
        print(f"Total row in DF                   : {len(category_df.index)}")
        
        # Merge DF
        new_topics_df = pd.concat([new_topics_df, category_df], ignore_index=True)
        
        print("-------------------------------\n")
    # Normalize group to int
    new_topics_df['group'] = new_topics_df['group'].astype(int)
    
    return new_topics_df