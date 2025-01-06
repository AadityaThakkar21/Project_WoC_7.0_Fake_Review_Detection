import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import os

# Download necessary NLTK resources (one-time download)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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
  return lemmatizer.lemmatize(text)


def clean_data(file_path):

  # Read the CSV file into a pandas DataFrame
  df = pd.read_csv(file_path)

  # Handle missing values (optional: you can impute missing values instead of dropping rows)
  df_cleaned = df.dropna()

  # Remove duplicate rows
  df_cleaned = df_cleaned.drop_duplicates()

  # Handle irrelevant entries (optional: you can keep these entries if they provide some information)
  df_cleaned = df_cleaned[df_cleaned['text_'].notnull()]

  # Create a new column to store the cleaned text
  df_cleaned['cleaned_text'] = df_cleaned['text_'].apply(clean_text_data)

  # Tokenize the text (split into words)
  df_cleaned['tokens'] = df_cleaned['cleaned_text'].apply(word_tokenize)

  # Remove stopwords (common words like "the", "a", "an", "in")
  stop_words = set(stopwords.words('english'))

  def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

  df_cleaned['filtered_tokens'] = df_cleaned['tokens'].apply(remove_stopwords)

  # Get the directory where the CSV file is located
  output_directory = os.path.dirname(file_path)

  # Create a filename for the cleaned data file
  cleaned_file_path = os.path.join(output_directory, 'cleaned_' + os.path.basename(file_path))

  # Save the cleaned data to a new CSV file
  df_cleaned.to_csv(cleaned_file_path, index=False)
  print(f"Cleaned data saved to: {cleaned_file_path}")


# Example usage
file_path = 'dataset/fakeReviewData.csv'  # Replace with the path to your CSV file
clean_data(file_path)