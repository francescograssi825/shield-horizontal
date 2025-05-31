import argparse
from strategies import standard, kfold, stratified_kfold

def main():
    parser = argparse.ArgumentParser(description="Train a binary classifier using CodeT5")
    parser.add_argument('--strategy', type=str, default='standard', 
                        choices=['standard', 'kfold', 'stratified_kfold'],
                        help='Validation strategy')
    parser.add_argument('--k', type=int, default=5, help='Number of folds for k-fold strategies')
    parser.add_argument('--model', type=str, default="Salesforce/codet5-small", help='Model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--data_path', type=str, default="../datasets/training_dataset.csv", help='Path to dataset')
    
    args = parser.parse_args()

    if args.strategy == 'standard':
        standard.run(args)
    elif args.strategy == 'kfold':
        kfold.run(args)
    elif args.strategy == 'stratified_kfold':
        stratified_kfold.run(args)

if __name__ == "__main__":
    main()