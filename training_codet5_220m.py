import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    T5ForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

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

# Metrics function - CORREZIONE per T5
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # T5 restituisce predictions come tupla, prendiamo il primo elemento
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convertiamo a numpy array se necessario
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # T5 può restituire predictions con shape complesso
    # Prendiamo solo la prima dimensione se necessario
    if len(predictions.shape) > 2:
        predictions = predictions[0]
    
    # Se le predictions sono ancora problematiche, le flattiamo
    if predictions.shape[0] != len(labels):
        predictions = predictions.reshape(-1, predictions.shape[-1])[:len(labels)]
    
    predictions = np.argmax(predictions, axis=1)
    
    # Assicuriamoci che labels e predictions abbiano la stessa lunghezza
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

def main():
    # Configuration - CodeT5-220M
    MODEL_NAME = "Salesforce/codet5-small"  # 220M parametri
    CSV_FILE = "training_dataset.csv"  # Sostituisci con il tuo file CSV
    MAX_LENGTH = 256  # CodeT5 funziona bene con sequenze più corte
    BATCH_SIZE = 8    # Possiamo usare batch size più grande con CodeT5
    LEARNING_RATE = 5e-5  # Learning rate leggermente più alto per T5
    NUM_EPOCHS = 3
    OUTPUT_DIR = "./codet5_binary_classifier"
    
    # Configurazioni PyTorch per ottimizzare memoria
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Libera cache GPU se disponibile
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU disponibile: {torch.cuda.get_device_name()}")
        print(f"Memoria GPU libera: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
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
    print("Caricamento del tokenizer e del modello CodeT5...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # CodeT5 ha già un pad_token, ma verifichiamo
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carica il modello T5 per classificazione binaria (2 classi)
    model = T5ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Imposta pad_token_id nel modello
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Modello caricato: {MODEL_NAME}")
    print(f"Parametri del modello: {model.num_parameters():,}")
    
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
    
    # Training arguments ottimizzati per CodeT5-220M
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        
        # BATCH SIZE - CodeT5 è più leggero, possiamo usare batch più grandi
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # GRADIENT ACCUMULATION - Meno necessario con CodeT5
        gradient_accumulation_steps=2,
        
        # OTTIMIZZAZIONI MEMORIA
        dataloader_pin_memory=True,  # Possiamo abilitare con CodeT5
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),  # Mixed precision se GPU disponibile
        gradient_checkpointing=True,     # Sempre utile per risparmiare memoria
        
        # LEARNING RATE
        learning_rate=LEARNING_RATE,
        
        # SCHEDULER E OTTIMIZZAZIONI
        warmup_steps=200,
        weight_decay=0.01,
        
        # LOGGING E SAVING
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,    # Più frequente per monitorare
        
        save_strategy="steps", 
        save_steps=200,
        save_total_limit=3,  # Mantieni 3 checkpoint migliori
        
        # METRICHE
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # DISABILITA REPORT ESTERNI
        report_to=None,
        
        # OTTIMIZZAZIONI AGGIUNTIVE
        max_grad_norm=1.0,
        optim="adamw_torch" if torch.cuda.is_available() else "adamw_torch",
        
        # SEED per riproducibilità
        seed=42,
        data_seed=42,
    )
    
    # Configurazione specifica per T5
    model.config.use_cache = False  # Disabilita cache per training
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Usa processing_class invece di tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Inizio del training...")
    print(f"Steps totali: {len(train_dataset) // (BATCH_SIZE * training_args.gradient_accumulation_steps) * NUM_EPOCHS}")
    
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
    test_text = "def calculate_sum(a, b): return a + b"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    print(f"Testo: '{test_text}'")
    print(f"Predizione: {predicted_class} ({model.config.id2label[predicted_class]})")
    print(f"Confidenza: {confidence:.4f}")
    
    # Stampa statistiche finali
    if torch.cuda.is_available():
        print(f"\nMemoria GPU utilizzata: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()