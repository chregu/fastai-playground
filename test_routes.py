import random
import time
from fastai.learner import load_learner
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import joblib

# Database connection
start_time = time.time()
from sqlalchemy import create_engine
connection_string = "postgresql://postgres:postgres@localhost:54322/postgres"
engine = create_engine(connection_string)
db_connect_time = time.time() - start_time

# Load the saved model and label encoder
print("Loading model...")
model_start = time.time()
learn = load_learner('route_classifier_model.pkl')
encoder = SentenceTransformer('BAAI/bge-m3')
label_encoder = joblib.load('route_label_encoder.pkl')
model_load_time = time.time() - model_start

# Get some random test cases from DB
print("Loading random test cases...")
query_start = time.time()
test_df = pd.read_sql("""
    SELECT query, route 
    FROM zuericitygpt_routes
 --   WHERE  route = 'promptinjection' 
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
    query_emb = encoder.encode(row['query'])
    embedding_time = time.time() - embed_start
    total_embedding_time += embedding_time

    # Create a DataFrame with the embeddings using the same column names as training
    embed_cols = [f'embed_{i}' for i in range(len(query_emb))]
    feature_df = pd.DataFrame([query_emb], columns=embed_cols)

    # Get prediction
    infer_start = time.time()
    dl = learn.dls.test_dl(feature_df)
    pred = learn.get_preds(dl=dl)[0]
    predicted_class_idx = pred[0].argmax().item()
    predicted_route = label_encoder.inverse_transform([predicted_class_idx])[0]
    inference_time = time.time() - infer_start
    total_inference_time += inference_time

    # Get prediction probabilities for top 3 classes
    top_3_indices = pred[0].argsort(descending=True)[:3]
    top_3_probs = pred[0][top_3_indices]
    top_3_routes = label_encoder.inverse_transform(top_3_indices)

    print(f"Query: {row['query'][:100]}...")
    print(f"Actual route: {row['route']}")
    print(f"Predicted route: {predicted_route}")
    print("\nTop 3 predictions:")
    for route, prob in zip(top_3_routes, top_3_probs):
        print(f"{route}: {prob:.2%}")
    print(f"Correct: {'✓' if predicted_route == row['route'] else '✗'}")
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