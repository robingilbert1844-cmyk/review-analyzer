import os
import re
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load vectorizer and primary classifier
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "sgd_classifier.pkl")
LDA_PATH = os.path.join("models", "lda_topic_model.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

# Load optional LDA complaint extractor
lda_model, lda_vec = joblib.load(LDA_PATH) if os.path.exists(LDA_PATH) else (None, None)

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'for', 'from',
    'had', 'has', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'me', 'more', 'most', 'my', 'no', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'out', 'over', 'own',
    'so', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'why', 'with',
    'you', 'your'
}

def clean_input_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    complaint_theme = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("review_text", "").strip()
        cleaned = clean_input_text(input_text)

        if cleaned:
            vec_input = vectorizer.transform([cleaned])
            prediction = model.predict(vec_input)[0]

            # If Negative, diagnose operational complaint theme via LDA
            if prediction == "Negative" and lda_model is not None:
                dtm_input = lda_vec.transform([cleaned])
                topic_id = lda_model.transform(dtm_input).argmax()
                theme_labels = {
                    0: "Overall Experience & Expectation Gap",
                    1: "Performance & Usability Failure",
                    2: "Return / Flipkart Service Issue",
                    3: "Defective Build / Poor Quality",
                    4: "Value for Money / Dead on Arrival"
                }
                complaint_theme = theme_labels.get(topic_id, "General Defect")

    return render_template("index.html", 
                           prediction=prediction, 
                           complaint_theme=complaint_theme, 
                           review_text=input_text)

if __name__ == "__main__":
    app.run(debug=True, port=5000)