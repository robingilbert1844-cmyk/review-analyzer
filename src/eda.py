import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

def run_eda(input_csv="data/cleaned_reviews.csv", output_dir="outputs"):
    print("--- STEP 4: Running Exploratory Data Analysis (EDA) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv)
    sns.set_theme(style="whitegrid")
    
    # 1. Visualization Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Rating Distribution
    sns.countplot(data=df, x='rating', palette='viridis', ax=axes[0])
    axes[0].set_title('Star Rating Distribution (1 to 5 Stars)')
    axes[0].set_xlabel('Star Rating')
    axes[0].set_ylabel('Review Count')

    # Plot 2: Sentiment Class Distribution
    sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'], palette='Set2', ax=axes[1])
    axes[1].set_title('Sentiment Distribution (Target Category)')
    axes[1].set_xlabel('Sentiment')
    axes[1].set_ylabel('Review Count')

    # Plot 3: Word Count Distribution
    df['word_count'] = df['cleaned_review'].apply(lambda x: len(str(x).split()))
    sns.histplot(df['word_count'], bins=30, kde=True, color='teal', ax=axes[2])
    axes[2].set_title('Review Word Count Distribution')
    axes[2].set_xlabel('Word Count')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'eda_charts.png')
    plt.savefig(chart_path)
    print(f"Saved EDA plots to '{chart_path}'")

    # 2. Complaint Theme Discovery (WordCloud)
    neg_reviews = df[df['sentiment'] == 'Negative']['cleaned_review'].dropna()
    negative_text = " ".join(neg_reviews.head(10000))

    if len(negative_text.strip()) > 0:
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(negative_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Negative Reviews (Complaint Discovery)', fontsize=14)
        wordcloud_path = os.path.join(output_dir, 'complaint_wordcloud.png')
        plt.savefig(wordcloud_path)
        print(f"Saved complaint WordCloud to '{wordcloud_path}'")

    # 3. Print Key Summary Statistics
    print("\n--- SUMMARY METRICS ---")
    print(f"Total Analyzed Reviews: {len(df):,}")
    print("\nSentiment Percentage Breakdown:")
    print((df['sentiment'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

if __name__ == "__main__":
    run_eda()