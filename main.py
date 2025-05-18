# main.py (FastAPI backend)
from fastapi import FastAPI
from pydantic import BaseModel
from app.recommender import get_movie_recommendations
from app.query_parser import parse_query

app = FastAPI()

class UserInput(BaseModel):
    message: str

@app.post("/recommend")
def recommend(user_input: UserInput):
    prefs = parse_query(user_input.message)
    results = get_movie_recommendations(prefs)
    print(results)
    if not 'No matching movies found' in results:
        return {"result": True,"recommended_movies": results}
    else:
        return {"recommended_movies": results}