import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("data/bollywood_movies.csv")
df["metadata"] = df["title"] + " " + df["genres"] + " " + df["tags"] + " " + df["mood"]

# Build or load FAISS index
embeddings = model.encode(df["metadata"].tolist(), show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


def get_movie_recommendations(prefs):
    query = " ".join(filter(None, [prefs.get("genre"), prefs.get("mood"), prefs.get("similar_to"), str(prefs.get("year"))]))
    query_vec = model.encode([query])
    scores, indices = index.search(np.array(query_vec), k=3)

    results = []
    if len(indices) == 0:
        return ["No matching movies found."]
    else:
        for idx in indices[0]:
            title = df.iloc[idx]["title"]
            genres = df.iloc[idx]["genres"]
            results.append({'title':f"{title}", 'genres': ", ".join(genres.split('|'))})
        return results