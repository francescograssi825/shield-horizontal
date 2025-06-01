from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import random

# Usa il tokenizer di GraphCodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Carica un dataset (es: code_search_net)
dataset = load_dataset("code_search_net", "python", split='train[:1%]')

# Funzione per aggiungere una label dummy (es. 0 o 1 casuale)
def add_dummy_label(example):
    example["labels"] = random.randint(0, 1)  # binary classification dummy labels
    return example

dataset = dataset.map(add_dummy_label)

# Preprocessamento
def tokenize_function(example):
    return tokenizer(
        example["func_documentation_string"],
        example["func_code_string"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Imposta il formato PyTorch e specifica le colonne che userai
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Carica il modello per classificazione binaria
model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)

# Parametri di addestramento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    # evaluation_strategy="epoch",  # commenta o rimuovi questa linea se la versione è vecchia
    # evaluate_during_training=True,  # alternativa nelle versioni precedenti (se supportato)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Avvia il training
trainer.train()
