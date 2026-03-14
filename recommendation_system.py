# Content-based movie recommendation system
# Developed as a machine learning project
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data(filepath):
    """Loads the Netflix dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def create_soup(x):
    """Combines metadata columns into a single string for analysis."""
    # Filling NaN with empty string
    title = str(x['title']) if pd.notna(x['title']) else ''
    director = str(x['director']) if pd.notna(x['director']) else ''
    cast = str(x['cast']) if pd.notna(x['cast']) else ''
    listed_in = str(x['listed_in']) if pd.notna(x['listed_in']) else ''
    description = str(x['description']) if pd.notna(x['description']) else ''
    
    # Concatenate all features
    # Weighting genres and description slightly more by adding them? 
    # For now, just a flat concatenation. The user mentioned TF-IDF on description/sentence embeddings.
    # User also laid out: TF-IDF + cosine similarity, Genre similarity matrices.
    # We will combine key text fields.
    return f"{title} {director} {cast} {listed_in} {description}"

def build_recommendation_model(df):
    """Builds the TF-IDF matrix and cosine similarity matrix."""
    print("Building recommendation model...")
    # Create a 'soup' column
    df['soup'] = df.apply(create_soup, axis=1)
    
    # TF-IDF Vectorizer
    # strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Construct the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # Compute the cosine similarity matrix
    # linear_kernel is equivalent to cosine_similarity for normalized vectors (TF-IDF is normalized)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim, indices):
    """Returns top 10 recommendations for a given title."""
    try:
        # Get the index of the movie that matches the title
        idx = indices[title]
        
        # Handle duplicate titles if necessary (taking the first one)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies (skipping the first one which is itself)
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df['title'].iloc[movie_indices]
    except KeyError:
        return [f"Title '{title}' not found in dataset."]

def main():
    filepath = 'c:/Users/ayoub/Desktop/data/netflix_titles_clean.csv'
    df = load_data(filepath)
    if df is None:
        return

    # Create indices for faster lookup
    # Drop duplicates in title just in case to ensure unique index
    df_unique = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    indices = pd.Series(df_unique.index, index=df_unique['title']).drop_duplicates()

    cosine_sim = build_recommendation_model(df_unique)

    # Interactive loop
    print("\nNetflix Recommendation System")
    print("-----------------------------")
    while True:
        title = input("\nEnter a movie title (or type 'quit' to exit): ").strip()
        if title.lower() == 'quit':
            break
        
        # Check if title exists roughly (case insensitive match could be improved but exact match is required by current logic)
        # Extending logic to find closest match could be a nice touch, but for now complying with plan.
        if title not in indices:
             # Try case-insensitive lookup
             matches = df_unique[df_unique['title'].str.lower() == title.lower()]
             if not matches.empty:
                 title = matches.iloc[0]['title']
        
        print(f"Finding recommendations for '{title}'...")
        results = get_recommendations(title, df_unique, cosine_sim, indices)
        
        if isinstance(results, list):
             print(f"Error: {results[0]}")
        else:
            print(f"\nTop 10 Recommendations for '{title}':")
            print(results.to_string(index=False))
            
    print("Goodbye!")

if __name__ == "__main__":
    main()
