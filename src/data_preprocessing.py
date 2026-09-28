import pandas as pd
import re
import os

# Standard English stopwords set (No NLTK dependency needed)
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m',
    'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t',
    'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t',
    'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

def preprocess_flipkart_dataset(input_csv="data/flipkart_product.csv", output_csv="data/cleaned_reviews.csv"):
    print("--- STEP 1: Loading Flipkart Dataset ---")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Could not find '{input_csv}'. Make sure flipkart_product.csv is inside the 'data/' directory.")
        
    df = pd.read_csv(input_csv, encoding='latin1')
    print(f"Loaded {len(df):,} raw review records.")

    # Convert rating to numeric
    df['rating'] = pd.to_numeric(df['Rate'], errors='coerce')
    df.dropna(subset=['rating', 'Summary'], inplace=True)

    print("\n--- STEP 2: Creating Target Sentiment Labels ---")
    def assign_sentiment(r):
        if r <= 2.0:
            return 'Negative'
        elif r == 3.0:
            return 'Neutral'
        else:
            return 'Positive'

    df['sentiment'] = df['rating'].apply(assign_sentiment)

    print("\n--- STEP 3: Cleaning Review Text ---")
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        cleaned = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
        return " ".join(cleaned)

    df['full_review'] = df['Review'].fillna('') + " " + df['Summary'].fillna('')
    df['cleaned_review'] = df['full_review'].apply(clean_text)

    # Drop empty reviews
    df = df[df['cleaned_review'].str.strip() != ""]

    # Select final columns and save
    final_cols = ['ProductName', 'Price', 'rating', 'sentiment', 'Review', 'Summary', 'cleaned_review']
    df_final = df[final_cols].rename(columns={'ProductName': 'product_name', 'Review': 'review_title', 'Summary': 'review_text'})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"\nPreprocessing Complete! Saved {len(df_final):,} cleaned reviews to '{output_csv}'.")

if __name__ == "__main__":
    preprocess_flipkart_dataset()