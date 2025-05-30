import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    MistralForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Custom Dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Configuration
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    CSV_FILE = "training_dataset.csv"  # Sostituisci con il tuo file CSV
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    OUTPUT_DIR = "./mistral_binary_classifier"
    
    # Load dataset
    print("Caricamento del dataset...")
    df = pd.read_csv(CSV_FILE)
    
    # Verifica che il dataset abbia le colonne corrette
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Il CSV deve avere le colonne 'Sentence' e 'Label'")
    
    # Preprocessing
    texts = df['Sentence'].tolist()
    labels = df['Label'].tolist()
    
    # Verifica che le label siano 0 o 1
    unique_labels = set(labels)
    if not unique_labels.issubset({0, 1}):
        raise ValueError("Le label devono essere 0 o 1")
    
    print(f"Dataset caricato: {len(texts)} esempi")
    print(f"Distribuzione classi: {pd.Series(labels).value_counts().to_dict()}")
    
    # Split train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Load tokenizer and model
    print("Caricamento del tokenizer e del modello...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Aggiungi padding token se non esiste
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carica il modello per classificazione binaria (2 classi)
    model = MistralForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Imposta pad_token_id nel modello
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create datasets
    print("Creazione dei dataset...")
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, MAX_LENGTH
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, MAX_LENGTH
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    
    # BATCH SIZE - Riduci drasticamente
    per_device_train_batch_size=1,  # Era BATCH_SIZE, ora 1
    per_device_eval_batch_size=1,   # Era BATCH_SIZE, ora 1
    
    # GRADIENT ACCUMULATION - Compensa il batch size piccolo
    gradient_accumulation_steps=4,  # Simula batch_size=8 con meno memoria
    
    # OTTIMIZZAZIONI MEMORIA
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    fp16=True,  # Forza mixed precision
    gradient_checkpointing=True,  # IMPORTANTE: salva memoria
    
    # LEARNING RATE - Aggiusta per gradient accumulation
    learning_rate=LEARNING_RATE * 0.5,  # Riduci un po' il learning rate
    
    # OTTIMIZZAZIONI AGGIUNTIVE
    warmup_steps=100,  # Era 500, riduciamo
    weight_decay=0.01,
    
    # LOGGING E SAVING - Riduci frequenza
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=50,  # Era 10, aumentiamo per risparmiare
    eval_strategy="steps",
    eval_steps=1000,   # Era 500, aumentiamo
    
    save_strategy="steps", 
    save_steps=1000,   # Era 500, aumentiamo
    save_total_limit=2,  # Mantieni solo 2 checkpoint
    
    # METRICHE
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # DISABILITA REPORT
    report_to=None,
    
    # OTTIMIZZAZIONI AGGIUNTIVE PER MEMORIA
    max_grad_norm=1.0,  # Gradient clipping
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # Optimizer più efficiente
)
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Inizio del training...")
    trainer.train()
    
    # Evaluate the model
    print("Valutazione finale...")
    eval_results = trainer.evaluate()
    print("Risultati di valutazione:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save the model
    print(f"Salvataggio del modello in {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completato!")
    
    # Test inference on a sample
    print("\nTest di inferenza:")
    test_text = "This is a test sentence"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    print(f"Testo: '{test_text}'")
    print(f"Predizione: {predicted_class} ({model.config.id2label[predicted_class]})")
    print(f"Confidenza: {confidence:.4f}")

if __name__ == "__main__":
    main()