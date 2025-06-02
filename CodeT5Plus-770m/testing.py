import torch
import argparse
import pandas as pd
import os
from datetime import datetime
from custom_libs.data_handler import DataHandler
from custom_libs.pii_data_loader import PIIDataLoader
from custom_libs.codeT5Plus_770m_model_trainer import CodeT5Plus_770m_ModelTrainer

### Constants
MODEL_NAME='google-bert/bert-base-uncased' # Model from Hugging Face
SAVE_FOLDER='results_test' # Path where to save the data

### Functions
def save_metrics_to_csv(metrics):
	"""Save evaluation metrics to CSV file"""
	# Create results directory if it doesn't exist
	os.makedirs(os.path.dirname(RESULTS_DIR), exist_ok=True)
	
	# Add timestamp to metrics
	metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	metrics['model_path'] = MODEL_PATH
	metrics['dataset_path'] = DATASET_PATH
	
	# Convert metrics to DataFrame
	df = pd.DataFrame([metrics])
	
	# Check if file exists to determine if we need headers
	file_exists = os.path.isfile(RESULTS_DIR)
	
	# Save to CSV (append if file exists, create new if not)
	df.to_csv(RESULTS_DIR, mode='a', header=not file_exists, index=False)
	print(f"Metrics saved to: {RESULTS_DIR}")

def load_and_evaluate_model(device):
	""" Load the pre-trained BERT model and evaluate it on the test dataset """
	# Initialize data handler and load the test dataset
	data_handler = DataHandler(DATASET_PATH)
	data_handler.load_data()
	data_handler.clean_data()

	# Extract test texts and labels
	test_texts, test_labels = data_handler.get_test_data()

	# Create data loader instance
	#	Note: Empty lists for train/val data since we're only testing
	data_loader = PIIDataLoader([], [], [], [], test_texts, test_labels)

	# Get the test data loader for batch processing
	test_loader = data_loader.get_specific_dataloader(mode='test')

	# Initialize the BERT model and load weights and optimizer state
	model_trainer = CodeT5Plus_770m_ModelTrainer(model_name=MODEL_NAME, save_folder=SAVE_FOLDER, device=device)
	model_trainer.load_model(MODEL_PATH, OPTIMIZER_PATH)

	# Evaluate the model on the test set and print metrics
	metrics = model_trainer.evaluate(test_loader)

	# Save metrics to CSV
	if metrics:
		save_metrics_to_csv(metrics)

	# Generate and display ROC curve for model performance visualization
	model_trainer.plot_roc_curve(test_loader)

	return metrics

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="BERT model evaluation script for PII detection testing")
	parser.add_argument('-r', '--save_folder', type=str, required=True, help='Output CSV file path for saving metrics (required)')
	parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model checkpoint (required)')
	parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to the test dataset (required)')
	parser.add_argument('-o', '--optimizer_path', type=str, required=True, help='Path to the optimizer checkpoint (required)')
	return parser.parse_args()

### Main
def main():
	args = arg_commandline()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	# Update paths from command line arguments
	global MODEL_PATH, DATASET_PATH, RESULTS_DIR, OPTIMIZER_PATH
	MODEL_PATH = args.model_path
	DATASET_PATH = args.dataset_path
	RESULTS_DIR = args.save_folder
	OPTIMIZER_PATH = args.optimizer_path

	metrics = load_and_evaluate_model(device)
	if metrics:
		# Save to the specified results path
		save_metrics_to_csv(metrics)

if __name__ == "__main__":
	main()