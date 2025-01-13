import pandas as pd
from fastai.text.all import *
from sentence_transformers import SentenceTransformer
import numpy as np
from fastai.tabular.all import *
from tqdm import tqdm

# Database connection
from sqlalchemy import create_engine
connection_string = "postgresql://postgres:postgres@localhost:54322/postgres"
engine = create_engine(connection_string)

# Load data
print("Loading data from database...")
df = pd.read_sql("SELECT question, answer, rating FROM _history where rating is not null order by created desc LIMIT 10000", engine)
print(f"Loaded {len(df)} records")

# Initialize encoder
encoder = SentenceTransformer('BAAI/bge-m3')

def get_combined_embedding(row):
    q_emb = encoder.encode(row['question'])
    a_emb = encoder.encode(row['answer'])
    return np.concatenate([q_emb, a_emb])

# Create embeddings with progress bar
print("Creating embeddings...")
embeddings = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    embeddings.append(get_combined_embedding(row))
X = np.stack(embeddings)

# Create embedding DataFrame more efficiently
embed_cols = [f'embed_{i}' for i in range(X.shape[1])]
embeddings_df = pd.DataFrame(X, columns=embed_cols)

# Combine original dataframe with embeddings
df = pd.concat([df, embeddings_df], axis=1)

print("Preparing data loaders...")
# Split data and create DataLoaders
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(df,
                   procs=[Normalize],
                   cat_names=[],
                   cont_names=embed_cols,
                   y_names=['rating'],
                   splits=splits)

dls = to.dataloaders(bs=128)

# Create and train model with metrics
learn = tabular_learner(dls,
                        layers=[200,100],
                        metrics=[rmse, mae])

print("\nTraining model...")
learn.fit_one_cycle(15, 1e-3)

# Print final metrics
print("\nFinal metrics:")
valid_metrics = learn.validate()
print(f"RMSE (Root Mean Square Error): {valid_metrics[1]:.4f}")
print(f"MAE (Mean Absolute Error): {valid_metrics[2]:.4f}")



# Save the model
print("\nSaving model...")
learn.export('quality_model.pkl')



print("Done!")