# Project_WoC_7.0_Fake_Review_Detection

## Checkpoint 1: Data Cleaning and Preprocessing

#### 1. Text Cleaning:
- Converts all text to lowercase.
- Removes non-alphabetic characters and unnecessary spaces.
- Uses lemmatization to reduce words to their dictionary forms for meaningful analysis.

#### 2. Tokenization and Stopword Removal:
- Splits text into individual words (tokens).
- Removes commonly used stopwords (e.g., "the", "and") to focus on meaningful words.

#### 3. Data Deduplication and Validation:
- Drops duplicate and null rows to ensure the dataset is clean and ready for processing.

#### 4. TF-IDF Transformation:
- Converts cleaned text data into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF).

#### 5. File Handling:
- Processes the input CSV file and saves a cleaned version of the data as a new file.

---

### Technologies

- **Python 3.x**
- Libraries:
  - `pandas` for data manipulation.
  - `nltk` for natural language processing.
  - `scikit-learn` for TF-IDF vectorization.

---

### Setup

1. **Install the required libraries**:
   ```bash
   pip install pandas nltk scikit-learn

2. **Download necessary NLTK resources**:

    These are automatically downloaded when you run the script.
