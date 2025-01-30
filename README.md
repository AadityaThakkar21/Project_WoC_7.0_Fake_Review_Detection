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

## Checkpoint 2: Model Training and Evaluation

#### 1. Data Loading:
- Loads the cleaned dataset produced in Checkpoint 1 from a CSV file.
- Validates that the dataset contains the required `cleaned_text` and `label` columns and removes rows with missing values.
- Allows sampling a fraction of the dataset for quicker testing during development.

#### 2. Data Splitting:
- Splits the dataset into training (80%) and testing (20%) sets.

#### 3. Pipeline Creation:
- Defines machine learning pipelines for three models:
  - **Support Vector Machine (SVM)**: Uses `LinearSVC` with TF-IDF features.
  - **Logistic Regression**: Employs `LogisticRegression` with TF-IDF features.
  - **Random Forest**: Utilizes `RandomForestClassifier` with TF-IDF features.

#### 4. Model Training and Evaluation:
- Trains each pipeline on the training data.
- Evaluates models on the test set using accuracy and a classification report.
- Identifies the best-performing model based on accuracy.

#### 5. Model Saving:
- Saves the trained models as `.joblib` files for later use.
- Stores models in a dedicated directory (`../models/`).

#### 6. Model Testing:
- Loads saved models and tests their predictions on sample texts (e.g., "This product is amazing!", "Worst purchase ever. Do not buy!").

---

### Technologies

- **Python 3.x**
- Libraries:
  - `pandas` for data manipulation.
  - `nltk` for natural language processing.
  - `scikit-learn` for machine learning pipelines and evaluation.
  - `joblib` for saving and loading trained models.

---

### Setup

1. **Install the required libraries**:
   ```bash
   pip install pandas nltk scikit-learn joblib
   ```

2. **Download necessary NLTK resources**:
   - Resources are automatically downloaded when you run the script.

3. **Run the script**:
   - Ensure the cleaned dataset is located in the specified directory (`../checkpoint 1/dataset/cleaned_fakeReviewData.csv`).
   - Execute the script to train models and evaluate performance.

   ```bash
   python script_name.py
   ```

4. **Testing Saved Models**:
   - Use the script’s testing functionality to validate saved models on new sample data.

