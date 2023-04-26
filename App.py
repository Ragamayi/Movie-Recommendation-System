from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
import numpy as np

# Read in the movie dataset
data1 = pd.read_csv("1000 movie list.csv")

# Create a TfidfVectorizer object
tfidf = TfidfVectorizer(stop_words="english")

# Fit the TfidfVectorizer to the movie descriptions
tfidf.fit(data1["Description"].values.astype("U"))

# Transform the movie descriptions into TF-IDF vectors
tfidf_desc_matrix = tfidf.transform(data1["Description"].values.astype("U"))

# Create a TfidfVectorizer object for the movie genres
tfidf_genres = TfidfVectorizer(stop_words="english")
genres_str = data1["Genre"].fillna('').apply(lambda x: ' '.join(x.split('|')))
tfidf_genres_matrix = tfidf_genres.fit_transform(genres_str)

# Concatenate the two TF-IDF matrices
tfidf_matrix = pd.concat([pd.DataFrame(tfidf_desc_matrix.todense()), pd.DataFrame(tfidf_genres_matrix.todense())], axis=1)



# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get the indices and movie titles
indices = pd.Series(data1.index, index=data1["Movie name"]).drop_duplicates().to_dict()


@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/recommendation', methods=['POST'])
def movie_recommendations():

    title = request.form['movie_name']
    num_rec = int(request.form['num_rec'])

    
    # Check if the movie is in the indices dictionary
    if title not in indices.keys():
        return render_template('recommend.html', error=True, movie_name=title)

    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the cosine similarity scores of the movies with the target movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort the movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top most similar movies
    sim_scores = sim_scores[1:num_rec+1]
    
    # Get the indices of the top movies
    movie_indices = np.array([i[0] for i in sim_scores])
    
    # Get the names of the top movies
    recommendations = data1["Movie name"].iloc[movie_indices].tolist()

   
    links = []
    for name in recommendations:
        query = f"{name} movie"
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        link = f'<a href="{url}" target="_blank">{name}</a>'
        links.append(link)

    # Join the links into a single string
    links_html = "<br>".join(links)

    
    # Render the recommendation page with the buttons
    return render_template('recommend.html', recommendations=links_html)
app.run(debug = True)


