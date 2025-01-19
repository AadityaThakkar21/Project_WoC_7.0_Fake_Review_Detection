import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Loading the cleaned csv file from checkpoint 1
def load_dataset(file_path, sample_frac=None):
    try:
        df = pd.read_csv(file_path)

        # Ensure that 'cleaned_text' and 'label' columns exist and remove rows with any missing values
        if 'cleaned_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'cleaned_text' and 'label' columns.")
        
        # Remove rows with NaN in 'cleaned_text' or 'label'
        df = df.dropna(subset=['cleaned_text', 'label'])
        
        # Use just 10% of the data for quicker testing
        # Loading the entire dataset for training and testing is time consuming, so we reduce it
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"Reduced Dataset Shape: {df.shape}")
        
        print("Dataset loaded successfully.")
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

# Split data into training and testing sets
def split_data(df):
    X = df['cleaned_text'] 
    y = df['label'] 
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Defining pipelines
def create_pipelines():
    pipelines = {
        "SVM": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('model', LinearSVC(random_state=42, max_iter=1000))  
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('model', RandomForestClassifier(random_state=42, n_estimators=100))
        ])
    }
    return pipelines

# Train and Evaluate Models
def train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    results = {}
    for model_name, pipeline in pipelines.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)  # Train the model
        y_pred = pipeline.predict(X_test)  # Predict on test data
        
        # Performance evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Evaluation for {model_name}:")
        print(classification_report(y_test, y_pred))
        results[model_name] = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    return results

# Saving the models
def save_models(pipelines, output_dir="../models/"):
    os.makedirs(output_dir, exist_ok=True)
    for model_name, pipeline in pipelines.items():
        model_file = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_model.joblib")
        joblib.dump(pipeline, model_file)
        print(f"{model_name} saved as: {model_file}")

# Test the saved models on a sample data
def test_saved_models(sample_texts, output_dir="../models/"):
    print("\nTesting saved models on sample data...")
    for model_name in ["SVM", "Logistic Regression", "Random Forest"]:
        model_file = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_model.joblib")
        if os.path.exists(model_file):
            print(f"\nTesting saved model: {model_name}")
            saved_model = joblib.load(model_file)  
            predictions = saved_model.predict(sample_texts) 
            for text, pred in zip(sample_texts, predictions):
                print(f"Text: '{text}' -> Predicted Label: {pred}")
        else:
            print(f"Model file for {model_name} not found.")

def main():
    # File path to dataset
    input_file = '../checkpoint 1/dataset/cleaned_fakeReviewData.csv'
    
    # Load Dataset
    df = load_dataset(input_file, sample_frac=0.1)  # Use 10% of the dataset for quicker testing
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    print("Data successfully split into training and testing sets.")
    print(f"Training Set Size: {X_train.shape[0]}, Testing Set Size: {X_test.shape[0]}")
    
    # Create Pipelines
    pipelines = create_pipelines()
    
    # Train and Evaluate Models
    results = train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test)
    
    # Identify Best Model
    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]['accuracy']:.2f}")
    
    # Save Models
    save_models(pipelines)
    
    # Test Predictions Using Saved Models
    sample_texts = ["This product is amazing!", "Worst purchase ever. Do not buy!"]
    test_saved_models(sample_texts)

if __name__ == "__main__":
    main()
