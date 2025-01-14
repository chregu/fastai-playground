import requests
import time
from fastai.learner import load_learner
from sentence_transformers import SentenceTransformer
import pandas as pd
import joblib
import json
import re
from sqlalchemy import text
from dotenv import load_dotenv
import os
import aiohttp
import asyncio

# Load environment variables from .env file
load_dotenv()

# Get environment variables
org = os.getenv('ORG')
api_key = os.getenv('API_KEY')
base_url = os.getenv('BASE_URL')

# Database connection
start_time = time.time()
from sqlalchemy import create_engine
connection_string = "postgresql://postgres:postgres@localhost:54322/postgres"
engine = create_engine(connection_string)
db_connect_time = time.time() - start_time

async def async_update_metadata(session, org: str, history_id: str, metadata: dict, base_url: str):
    if not base_url:
        return False

    url = f"{base_url}/{org}/admin/history/metadata/{history_id}"
    try:
        async with session.patch(url, json=metadata, headers={"API-Key": api_key}) as response:
            return response.status == 200
    except Exception as e:
        print(f"Failed to update metadata for ID {history_id}: {str(e)}")
        return False




# Load the saved model and label encoder
print("Loading model...")
model_start = time.time()
learn = load_learner(f'{org}_route_classifier_model.pkl')
encoder = SentenceTransformer('BAAI/bge-m3')
label_encoder = joblib.load(f'{org}_route_label_encoder.pkl')
model_load_time = time.time() - model_start

# Get some random test cases from DB
print("Loading random test cases...")
query_start = time.time()
test_df = pd.read_sql(f"""
    SELECT id, question, route, metadata
    FROM _history
    WHERE metadata->>'modelClassifier' IS NULL
     AND organization = '{org}'
    -- ORDER BY RANDOM()
    ORDER BY created DESC
    -- LIMIT 100000
    
""", engine)
query_time = time.time() - query_start

def clean_question(question):
    # Remove /mode:.*/ pattern
    return re.sub(r'([a-zA-Z]+:\S*|^[A-Z_]+: )', '', question).strip()

print(f"\nSetup times:")
print(f"Database connection: {db_connect_time:.2f}s")
print(f"Model loading: {model_load_time:.2f}s")
print(f"Query execution: {query_time:.2f}s")

print("\nTesting predictions and updating metadata...")
print("-" * 80)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def process_rows(test_df):
    async with aiohttp.ClientSession() as session:
        tasks = []
        batch_size = 10
        total_inference_time = 0
        total_embedding_time = 0
        correct_predictions = 0
        total_samples = len(test_df)
        start_processing_time = time.time()

        for idx, row in enumerate(test_df.iterrows(), 1):
            _, row = row  # Unpack the row

            # Calculate and display progress
            progress = (idx / total_samples) * 100
            elapsed_time = time.time() - start_processing_time
            avg_time_per_sample = elapsed_time / idx
            estimated_remaining = avg_time_per_sample * (total_samples - idx)



            # Clean the question
            cleaned_question = clean_question(row['question'])

            # Get embeddings
            embed_start = time.time()
            query_emb = encoder.encode(cleaned_question, show_progress_bar=False)
            embedding_time = time.time() - embed_start
            total_embedding_time += embedding_time

            # Create a DataFrame with the embeddings
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

            # Update accuracy tracking
            if predicted_route == row['route']:
                correct_predictions += 1

            # Update metadata
            try:
                current_metadata = row['metadata'] if row['metadata'] else {}
                if isinstance(current_metadata, str):
                    current_metadata = json.loads(current_metadata)

                # Get prediction probabilities for top 3 classes
                top_3_indices = pred[0].argsort(descending=True)[:3]
                top_3_probs = pred[0][top_3_indices]
                top_3_routes = label_encoder.inverse_transform(top_3_indices)
                current_metadata['modelClassifier'] = {}
                current_metadata['modelClassifier']['route']= predicted_route

                # Create a list of dictionaries for the top 3 predictions
                top_3_predictions = [
                    {
                        "route": route,
                        "prob": round(float(prob), 2)   # Convert tensor to float for JSON serialization
                    }
                    for route, prob in zip(top_3_routes, top_3_probs)
                    if float(prob) >= 0.05
                ]
                current_metadata['modelClassifier']['prob'] = top_3_predictions
                if base_url:
                    # Fire and forget the request
                    task = asyncio.create_task(
                        async_update_metadata(
                            session,
                            org,
                            row['id'],
                            {'modelClassifier': current_metadata['modelClassifier']},
                            base_url
                        )
                    )
                    tasks.append(task)

                    # Process in batches
                    if len(tasks) >= batch_size:
                        await asyncio.gather(*tasks)
                        tasks = []
                # Update database
                update_query = text("""
                            UPDATE _history 
                            SET metadata = :metadata 
                            WHERE id = :id
                        """)

                with engine.connect() as conn:
                    conn.execute(update_query, {
                        'metadata': json.dumps(current_metadata),
                        'id': row['id']
                    })
                    conn.commit()

                # Only print detailed logs every 1000 samples to reduce output
                if idx % 10 == 0:
                    print(f"\nSample {idx}:")
                    print(f"Query: {cleaned_question}")
                    print(f"Actual route: {row['route']}")
                    print(f"Predicted route: {predicted_route}")
                    print("Top 3 predictions:")
                    for route, prob in zip(top_3_routes, top_3_probs):
                        print(f"{route}: {prob:.2%}")
                    print(f"Correct: {'✓' if predicted_route == row['route'] else '✗'}")
                    print(f"Embedding time: {embedding_time:.3f}s")
                    print(f"Inference time: {inference_time:.3f}s")
                    print(f"Progress: {progress:.1f}% ({idx}/{total_samples}) | "
                          f"Elapsed: {elapsed_time:.1f}s | "
                          f"ETA: {estimated_remaining:.1f}s | "
                          f"Accuracy: {(correct_predictions/idx)*100:.1f}%")
                    print("-" * 80)

            except Exception as e:
                print(f"\nError updating metadata for ID {row['id']}: {str(e)}")

        if tasks:
            await asyncio.gather(*tasks)
        # Print final summary
        total_time = time.time() - start_time
        avg_embedding_time = total_embedding_time / total_samples
        avg_inference_time = total_inference_time / total_samples
        final_accuracy = (correct_predictions / total_samples) * 100

        print(f"\n\nFinal Summary:")
        print(f"Processed {total_samples} samples in {total_time:.2f}s")
        print(f"Average embedding time per sample: {avg_embedding_time:.3f}s")
        print(f"Average inference time per sample: {avg_inference_time:.3f}s")
        print(f"Final accuracy: {final_accuracy:.2f}%")
        print(f"Total runtime: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(process_rows(test_df))