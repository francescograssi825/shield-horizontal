import os
from config import constants
from data.loader import load_data, split_data
from data.dataset import TextClassificationDataset
from model.model_loader import load_tokenizer, load_model
from model.trainer import configure_training, create_trainer, train_model, save_model
from utils.saver import save_results, save_training_args, save_roc_curve
from config.args_handler import process_missing_args

def run(args):
    args = process_missing_args(args)
    args.output_dir = os.path.join(constants.OUTPUT_DIRS['standard'], args.model.replace("/", "_"))
    os.makedirs(args.output_dir, exist_ok=True)
    
     #save training args for reproducibility
    save_training_args(args, args.output_dir)

    # Load and prepare data
    texts, labels = load_data(args.data_path)
    train_texts, val_texts, train_labels, val_labels = split_data(texts, labels)
    
    # Load model
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, constants.MODEL_CONFIG)
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    # Configure and run training
    training_args = configure_training(args)
    trainer = create_trainer(model, training_args, train_dataset, val_dataset, tokenizer)
    results = train_model(trainer)
    
    # Save results and model
    save_results('standard', results, args.output_dir)
    save_model(trainer, os.path.join(args.output_dir, 'model'))


    # save roc curve if is possible
    if 'y_true' in results and 'y_probs' in results:
        roc_data = save_roc_curve(results['y_true'], results['y_probs'], args.output_dir)
        results['roc_curve'] = roc_data  
    
    
    return results