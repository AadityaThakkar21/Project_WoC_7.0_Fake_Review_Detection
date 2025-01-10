import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean text
def clean_text_data(text):
    """
    This function cleans text data by converting it to lowercase,
    removing non-alphabetic characters and spaces, and stemming or lemmatizing the words.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # Stemming reduces words to their root form (e.g., "running" becomes "run")
    # Lemmatization reduces words to their dictionary form (e.g., "better" becomes "good")

    """
    Lemmatization is a better option in this case due to the following two reasons:

    1) Fake reviews often contain nuanced language, and lemmatization can help
    preserve the meaningful base form of words.
    For example, "better" → "good" is crucial in some contexts, as it helps maintain the meaning.

    2) Lemmatization avoids overly reducing words to roots that might not convey meaning
    or context correctly, making it more appropriate when trying to detect subtle patterns in the text.
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Function to process data
def clean_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    df = df.dropna().drop_duplicates()
    df = df[df['text_'].notnull()]
    
    # Clean text data
    df['cleaned_text'] = df['text_'].apply(clean_text_data)
    
    # Tokenize text
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['filtered_tokens'] = df['tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )
    
    # Save cleaned data
    output_directory = os.path.dirname(file_path)
    cleaned_file_path = os.path.join(output_directory, 'cleaned_' + os.path.basename(file_path))
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to: {cleaned_file_path}")
    return df

# Function to apply TF-IDF
def apply_tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    print("TF-IDF transformation completed.")
    return tfidf_matrix

# Example usage
file_path = 'dataset/fakeReviewData.csv'  # Replace with the path to your CSV file
df_cleaned = clean_data(file_path)

# Apply TF-IDF
tfidf_matrix = apply_tfidf(df_cleaned)
