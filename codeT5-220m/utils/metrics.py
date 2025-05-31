import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    if len(predictions.shape) > 2:
        predictions = predictions[0]
    
    if predictions.shape[0] != len(labels):
        predictions = predictions.reshape(-1, predictions.shape[-1])[:len(labels)]
    
    predictions = np.argmax(predictions, axis=1)
    
    # Ensure same length
    min_len = min(len(labels), len(predictions))
    labels = labels[:min_len]
    predictions = predictions[:min_len]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }