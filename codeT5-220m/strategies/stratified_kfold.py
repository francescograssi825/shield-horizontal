import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import constants
from data.loader import load_data
from data.dataset import TextClassificationDataset
from model.model_loader import load_tokenizer, load_model
from model.trainer import configure_training, create_trainer, train_model
from utils.saver import save_results, save_fold_results
from config.args_handler import process_missing_args

def run(args):
    args = process_missing_args(args)
    base_output_dir = os.path.join(constants.OUTPUT_DIRS['stratified_kfold'], args.model.replace("/", "_"))
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load data
    texts, labels = load_data(args.data_path)
    texts, labels = np.array(texts), np.array(labels)
    
    # StratifiedKFold setup
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n{'='*40}")
        print(f"Training stratified fold {fold+1}/{args.k}")
        print(f"{'='*40}")
        
        fold_dir = os.path.join(base_output_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        train_texts, train_labels = texts[train_idx].tolist(), labels[train_idx].tolist()
        val_texts, val_labels = texts[val_idx].tolist(), labels[val_idx].tolist()
        
        # Load model
        tokenizer = load_tokenizer(args.model)
        model = load_model(args.model, constants.MODEL_CONFIG)
        
        # Create datasets
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
        
        # Configure and run training
        training_args = configure_training(args, tokenizer)
        training_args.output_dir = fold_dir
        trainer = create_trainer(model, training_args, train_dataset, val_dataset, tokenizer)
        results = train_model(trainer)
        
        # Save fold results
        save_fold_results(fold+1, results, fold_dir)
        fold_results.append(results)
    
    # Save aggregated results
    save_results('stratified_kfold', aggregate_results(fold_results), base_output_dir)
    return fold_results

def aggregate_results(fold_results):
    aggregated = {}
    for metric in fold_results[0].keys():
        values = [res[metric] for res in fold_results]
        aggregated[f'avg_{metric}'] = np.mean(values)
        aggregated[f'std_{metric}'] = np.std(values)
    return aggregated