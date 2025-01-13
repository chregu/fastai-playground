import pandas as pd
from fastai.text.all import *
from sentence_transformers import SentenceTransformer
import numpy as np
from fastai.tabular.all import *
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Database connection
from sqlalchemy import create_engine
connection_string = "postgresql://postgres:postgres@localhost:54322/postgres"
engine = create_engine(connection_string)

# Load data
print("Loading data from database...")
df = pd.read_sql("SELECT query, route FROM zuericitygpt_routes where auto_detected = false limit 10000", engine)
print(f"Loaded {len(df)} records")

# Initialize encoder
encoder = SentenceTransformer('BAAI/bge-m3')

def get_query_embedding(row):
    return encoder.encode(row['query'])

# Create embeddings with progress bar
print("Creating embeddings...")
embeddings = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    embeddings.append(get_query_embedding(row))
X = np.stack(embeddings)

# Create embedding DataFrame
embed_cols = [f'embed_{i}' for i in range(X.shape[1])]
embeddings_df = pd.DataFrame(X, columns=embed_cols)

# Encode routes
label_encoder = LabelEncoder()
df['route_encoded'] = label_encoder.fit_transform(df['route'])

# Save label encoder classes for later use
route_classes = label_encoder.classes_
print(f"Route classes: {route_classes}")

# Combine original dataframe with embeddings
df = pd.concat([df, embeddings_df], axis=1)

print("Preparing data loaders...")
# Split data and create DataLoaders
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(df,
                   procs=[Normalize],
                   cat_names=[],  # No categorical variables now, we encoded them
                   cont_names=embed_cols,
                   y_names=['route_encoded'],  # Using encoded route
                   splits=splits)

dls = to.dataloaders(bs=128)

# Create and train model
n_classes = len(route_classes)
learn = tabular_learner(dls,
                        layers=[400,200],
                        n_out=n_classes,  # Specify number of output classes
                        loss_func=CrossEntropyLossFlat(),
                        metrics=accuracy)

print("\nTraining model...")
learn.fit_one_cycle(15, 1e-3)

# Print final metrics
print("\nFinal metrics:")
valid_metrics = learn.validate()
print(f"Accuracy: {valid_metrics[1]:.4f}")

# Save the model and label encoder
print("\nSaving model and encoder...")
learn.export('route_classifier_model.pkl')
import joblib
joblib.dump(label_encoder, 'route_label_encoder.pkl')

print("Done!")