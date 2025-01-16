import pandas as pd

# Loading the cleaned csv file from checkpoint 1
def load_dataset(file_path, sample_frac=None):
    try:
        df = pd.read_csv(file_path)

        # Ensure that 'cleaned_text' and 'label' columns exist and remove rows with missing values
        if 'cleaned_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'cleaned_text' and 'label' columns.")
        
        # Remove rows with NaN in 'cleaned_text' or 'label'
        df = df.dropna(subset=['cleaned_text', 'label'])
        
        # If specified, reduce the dataset size
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)  # Use 10% of the data for quicker testing
            print(f"Reduced Dataset Shape: {df.shape}")
        
        print("Dataset loaded successfully.")
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()


input_file = '../checkpoint 1/dataset/cleaned_fakeReviewData.csv'
    
# Load Dataset
df = load_dataset(input_file, sample_frac=0.1)  # Using only 10% of the dataset for quicker testing