import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    
    stop_words = set(stopwords.words('russian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    stemmer = SnowballStemmer("russian")
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return stemmed_tokens

input_text = "Ваш текст для обработки."
processed_text = preprocess_text(input_text)
print(processed_text)