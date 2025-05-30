import torch
from transformers import AutoTokenizer


class PIIDataLoader:
	""" A data loader class for handling Personally Identifiable Information (PII) detection datasets """
	
	def __init__(self, train_texts=None, train_labels=None, valid_texts=None, valid_labels=None, test_texts=None, test_labels=None, tokenizer_name='microsoft/longcoder-base', batch_size=8):
		""" Initialize the PIIDataLoader with text data and labels for different splits """
		self.train_texts = train_texts
		self.train_labels = train_labels
		self.valid_texts = valid_texts if valid_texts is not None else []
		self.valid_labels = valid_labels if valid_labels is not None else []
		self.test_texts = test_texts
		self.test_labels = test_labels
		self.batch_size = batch_size
		
		# Initialize the LONGCODERBASE tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	
	def tokenize_data(self, texts):
		""" Tokenize a list of text samples using the LONGCODERBASE tokenizer """

		if not texts:
			return None
		
		# Tokenize with truncation, padding, and return PyTorch tensors
		return self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=32) # Gabriele aggiunge max lenght
	
	def create_dataset(self, encodings, labels):
		""" Create a PyTorch Dataset from tokenized encodings and labels """

		return PIIDataset(encodings, labels)
	
	def get_specific_dataloader(self, mode):
		""" Get a specific DataLoader for a given mode (train, valid, or test) """

		if mode == 'train':
			# Tokenize training data
			train_encodings = self.tokenize_data(self.train_texts)
			train_dataset = PIIDataset(train_encodings, self.train_labels)
			# Return DataLoader with shuffling enabled for training
			return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		
		elif mode == 'valid':
			# Tokenize validation data
			valid_encodings = self.tokenize_data(self.valid_texts)
			valid_dataset = PIIDataset(valid_encodings, self.valid_labels)
			# Return DataLoader without shuffling for validation
			return torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
		
		elif mode == 'test':
			# Tokenize test data
			test_encodings = self.tokenize_data(self.test_texts)
			test_dataset = PIIDataset(test_encodings, self.test_labels)
			# Return DataLoader without shuffling for testing
			return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
		
		else:
			raise ValueError("mode should be 'train', 'valid', or 'test'")
	
	def get_dataloader(self):
		""" Get all DataLoaders (train, validation, test) at once """

		# Tokenize all datasets
		train_encodings = self.tokenize_data(self.train_texts)
		valid_encodings = self.tokenize_data(self.valid_texts)
		test_encodings = self.tokenize_data(self.test_texts)
		
		# Create PyTorch datasets
		train_dataset = PIIDataset(train_encodings, self.train_labels)
		valid_dataset = PIIDataset(valid_encodings, self.valid_labels)
		test_dataset = PIIDataset(test_encodings, self.test_labels)
		
		# Create DataLoaders
		# Training loader with shuffling enabled
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		# Validation loader without shuffling
		valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
		# Test loader without shuffling
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
		
		return train_loader, valid_loader, test_loader


class PIIDataset(torch.utils.data.Dataset):
	""" Custom PyTorch Dataset class for PII detection data """
	
	def __init__(self, encodings, labels):
		""" Initialize the dataset with encodings and labels """

		self.encodings = encodings
		self.labels = labels
	
	def __getitem__(self, idx):
		""" Get a single item from the dataset at the specified index """

		# Extract all tokenizer outputs (input_ids, attention_mask, etc.) for the given index
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		# Add the corresponding label
		item['labels'] = torch.tensor(self.labels[idx])
		return item
	
	def __len__(self):
		""" Get the total number of samples in the dataset """

		return len(self.labels)