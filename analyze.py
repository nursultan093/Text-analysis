import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]

    stop_words = set(stopwords.words('russian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    stemmer = SnowballStemmer("russian")
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return stemmed_tokens

def analyze_sentiment_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)

    if sentiment_score['compound'] >= 0.05:
        sentiment_category = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment_category = 'Negative'
    else:
        sentiment_category = 'Neutral'

    return sentiment_score, sentiment_category

def extract_keywords(text, num_keywords=5):
    tokens = word_tokenize(text)
    fdist = FreqDist(tokens)
    keywords = [word for word, freq in fdist.most_common(num_keywords)]
    
    return keywords

# Пример использования
input_text = "Этот продукт просто удивителен! Очень доволен покупкой."
processed_text = preprocess_text(input_text)
sentiment_score, sentiment_category = analyze_sentiment_nltk(input_text)
keywords = extract_keywords(input_text)

print(f"Тональность: {sentiment_category}")
print(f"Оценка тональности: {sentiment_score}")
print(f"Ключевые слова: {keywords}")
