def process_missing_args(args):
    """Handle missing arguments with default values"""
    defaults = {
        'k': 5,
        'model': "Salesforce/codet5-small",
        'epochs': 3,
        'batch_size': 8,
        'max_length': 256,
        'data_path': "../datasets/training_dataset.csv"
    }
    
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    return args