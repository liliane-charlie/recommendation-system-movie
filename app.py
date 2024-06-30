from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = FastAPI()

# Load your movies data
movies_data = pd.read_csv(r'C:\Users\USER\OneDrive\Desktop\movies\movies.csv')  # Ensure this path is correct

# Check if 'index' is in the DataFrame, if not create it
if 'index' not in movies_data.columns:
    movies_data.reset_index(inplace=True)

# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replacing the null values with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining the selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Creating the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Generating the feature vectors
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculating the cosine similarity
similarity = cosine_similarity(feature_vectors)

class MovieRequest(BaseModel):
    movie_name: str

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommendations/")
def get_recommendations(request: MovieRequest):
    movie_name = request.movie_name
    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        raise HTTPException(status_code=404, detail="Movie not found")

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index.values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies[1:31]:  # Skip the first one because it's the same movie
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommendations.append(title_from_index)

    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)