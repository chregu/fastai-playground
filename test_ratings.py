import random
import time
from fastai.learner import load_learner
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Database connection
start_time = time.time()
from sqlalchemy import create_engine
connection_string = "postgresql://postgres:postgres@localhost:54322/postgres"
engine = create_engine(connection_string)
db_connect_time = time.time() - start_time

# Load the saved model
print("Loading model...")
model_start = time.time()
learn = load_learner('quality_model.pkl')
encoder = SentenceTransformer('BAAI/bge-m3')
model_load_time = time.time() - model_start

# Get some random test cases from DB
print("Loading random test cases...")
query_start = time.time()
test_df = pd.read_sql("""
    SELECT question, answer, rating 
    FROM _history 
    WHERE rating IS NOT NULL 
    ORDER BY RANDOM() 
    LIMIT 10
""", engine)
query_time = time.time() - query_start

print(f"\nSetup times:")
print(f"Database connection: {db_connect_time:.2f}s")
print(f"Model loading: {model_load_time:.2f}s")
print(f"Query execution: {query_time:.2f}s")

print("\nTesting predictions...")
print("-" * 80)

total_inference_time = 0
total_embedding_time = 0

for _, row in test_df.iterrows():
    # Get embeddings
    embed_start = time.time()
    q_emb = encoder.encode(row['question'])
    a_emb = encoder.encode(row['answer'])
    combined = np.concatenate([q_emb, a_emb])
    embedding_time = time.time() - embed_start
    total_embedding_time += embedding_time

    # Create a DataFrame with the embeddings using the same column names as training
    embed_cols = [f'embed_{i}' for i in range(len(combined))]
    feature_df = pd.DataFrame([combined], columns=embed_cols)

    # Get prediction
    infer_start = time.time()
    dl = learn.dls.test_dl(feature_df)
    pred = learn.get_preds(dl=dl)[0]
    predicted_rating = pred[0].item()
    inference_time = time.time() - infer_start
    total_inference_time += inference_time

    print(f"Question: {row['question'][:100]}...")
    print(f"Answer: {row['answer'][:100]}...")
    print(f"Actual rating: {row['rating']}")
    print(f"Predicted rating: {predicted_rating:.1f}")
    print(f"Difference: {abs(row['rating'] - predicted_rating):.1f}")
    print(f"Embedding time: {embedding_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s")
    print("-" * 80)

avg_embedding_time = total_embedding_time / len(test_df)
avg_inference_time = total_inference_time / len(test_df)
total_time = time.time() - start_time

print(f"\nSummary:")
print(f"Average embedding time per sample: {avg_embedding_time:.3f}s")
print(f"Average inference time per sample: {avg_inference_time:.3f}s")
print(f"Total runtime: {total_time:.2f}s")