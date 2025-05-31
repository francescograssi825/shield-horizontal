# Output directories
OUTPUT_DIRS = {
    'standard': 'results/standard',
    'kfold': 'results/kfold',
    'stratified_kfold': 'results/stratified_kfold'
}

# Model configuration
MODEL_CONFIG = {
    'id2label': {0: "NEGATIVE", 1: "POSITIVE"},
    'label2id': {"NEGATIVE": 0, "POSITIVE": 1}
}