from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from recommendation_system import load_data, build_recommendation_model, get_recommendations

app = Flask(__name__)

# OMDb API Key
OMDB_API_KEY = "trilogy"

# Global variables
df_unique = None
cosine_sim = None
indices = None

def initialize_model():
    global df_unique, cosine_sim, indices
    print("Loading data and building model...")
    filepath = 'netflix_titles.csv'
    df = load_data(filepath)
    if df is not None:
        df_unique = df.drop_duplicates(subset=['title']).reset_index(drop=True)
        indices = pd.Series(df_unique.index, index=df_unique['title']).drop_duplicates()
        cosine_sim = build_recommendation_model(df_unique)
        print("Model built successfully.")
    else:
        print("Failed to load data.")

def fetch_poster(title):
    """Fetches the poster URL from OMDb API."""
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                return data.get('Poster')
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    search_query = ""
    error_message = None

    if request.method == 'POST':
        search_query = request.form.get('title', '').strip()
        if indices is None:
            error_message = "The recommendation model failed to load. Please check the server logs."
        elif search_query:
            query_title = search_query
            if search_query not in indices:
                 matches = df_unique[df_unique['title'].str.lower() == search_query.lower()]
                 if not matches.empty:
                     query_title = matches.iloc[0]['title']
            
            rec_titles = get_recommendations(query_title, df_unique, cosine_sim, indices)
            
            if isinstance(rec_titles, list):
                 error_message = rec_titles[0]
            else:
                rec_df = df_unique[df_unique['title'].isin(rec_titles)].copy()
                rec_df['title'] = pd.Categorical(rec_df['title'], categories=rec_titles, ordered=True)
                rec_df = rec_df.sort_values('title')
                
                # Fetch posters in parallel
                titles = rec_df['title'].tolist()
                with ThreadPoolExecutor(max_workers=10) as executor:
                    posters = list(executor.map(fetch_poster, titles))
                
                rec_df['poster'] = posters
                recommendations = rec_df.to_dict('records')

    return render_template('index.html', recommendations=recommendations, search_query=search_query, error_message=error_message)

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
