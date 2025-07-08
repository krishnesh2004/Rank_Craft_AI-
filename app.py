from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()
model, tfidf = pickle.load(open("rank_model.pkl", "rb"))

class PageData(BaseModel):
    desc: str

@app.post("/optimize")
def optimize(page: PageData):
    desc_len = len(page.desc.split())
    keyword_count = page.desc.lower().count('laptop')
    tfidf_vec = tfidf.transform([page.desc]).toarray()[0].tolist()
    features = tfidf_vec + [desc_len, keyword_count]
    prediction = model.predict([features])[0]
    return {"likely_to_rank_high": bool(prediction)}
