import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """Load and validate dataset"""
    df = pd.read_csv(data_path)
    
    # Validate columns
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        raise ValueError("CSV must contain 'Sentence' and 'Label' columns")
    
    # Validate labels
    if not set(df['Label']).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    
    return df['Sentence'].tolist(), df['Label'].tolist()

def split_data(texts, labels, test_size=0.2, stratify=True):
    """Split data into train and validation sets"""
    return train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels if stratify else None
    )