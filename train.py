import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('example_data.csv')

# Feature engineering
df['desc_len'] = df['desc'].apply(lambda x: len(x.split()))
df['has_laptop'] = df['desc'].apply(lambda x: 1 if 'laptop' in x.lower() else 0)

# TF-IDF on description
tfidf = TfidfVectorizer(max_features=10)
X_tfidf = tfidf.fit_transform(df['desc']).toarray()

# Combine all features
features = pd.concat(
    [pd.DataFrame(X_tfidf), df[['desc_len', 'has_laptop']].reset_index(drop=True)],
    axis=1
)

# Ensure columns are strings
features.columns = features.columns.astype(str)

# Train model
model = LogisticRegression()
model.fit(features, df['label'])

# Save model + vectorizer
with open("rank_model.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

print("✅ Model trained & saved as rank_model.pkl")
