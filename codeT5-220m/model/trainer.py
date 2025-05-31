import torch
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from utils.metrics import compute_metrics
from utils.logger import log_gpu_info

def configure_training(args):
    """Configure training arguments"""
    return TrainingArguments(
        output_dir=args.output_dir,                                  # Directory in cui salvare i checkpoint e gli output.
        num_train_epochs=args.epochs,                                # Numero totale di epoche per l'addestramento.
        per_device_train_batch_size=args.batch_size,                 # Dimensione del batch per dispositivo durante l'addestramento.
        per_device_eval_batch_size=args.batch_size,                  # Dimensione del batch per dispositivo durante la valutazione.
        gradient_accumulation_steps=2,                               # Numero di step per accumulare i gradienti prima di aggiornare i pesi.
        dataloader_pin_memory=True,                                  # Abilita il "pinning" della memoria nel dataloader per velocizzare il trasferimento dati.
        remove_unused_columns=False,                                 # Mantiene tutte le colonne dei dati, anche quelle non usate dal modello (utile per debugging o informazioni aggiuntive).
        fp16=torch.cuda.is_available(),                              # Abilita il training in precisione mista (16-bit) se è disponibile una GPU CUDA.
        gradient_checkpointing=True,                                 # Attiva il checkpointing dei gradienti per ridurre l’uso della memoria durante il training.
        learning_rate=5e-5,                                          # Tasso di apprendimento iniziale del modello.
        warmup_steps=200,                                            # Numero di step iniziali durante i quali il tasso di apprendimento aumenta gradualmente.
        weight_decay=0.01,                                           # Coefficiente di decadenza del peso per la regolarizzazione (L2).
        logging_dir=f'{args.output_dir}/logs',                       # Directory dove verranno salvati i log di addestramento.
        logging_steps=25,                                            # Frequenza (in step) con cui vengono registrati i log.
        eval_strategy="steps",                                       # Strategia per la valutazione: esegue l'eval a intervalli di step.
        eval_steps=200,                                              # Numero di step tra una valutazione e l'altra.
        save_strategy="steps",                                       # Strategia per salvare i checkpoint: salva a intervalli fissi di step.
        save_steps=200,                                              # Numero di step tra i salvataggi dei checkpoint.
        save_total_limit=3,                                          # Numero massimo di checkpoint da tenere, eliminando i più vecchi.
        load_best_model_at_end=True,                                 # Alla fine dell'allenamento, carica il modello che ha ottenuto i migliori risultati.
        metric_for_best_model="f1",                                  # Metrica utilizzata per determinare il "miglior" modello (qui l'F1 score).
        greater_is_better=True,                                      # Indica che, per la metrica scelta, valori più alti sono migliori.
        report_to=None,                                              # Disabilita il report automatico a strumenti esterni (come WandB o Comet).
        max_grad_norm=1.0,                                           # Norma massima per il clipping dei gradienti per evitare gradienti esplosivi.
        optim="adamw_torch" if torch.cuda.is_available() else "adamw_torch",  # Ottimizzatore da utilizzare (in entrambi i casi qui si usa "adamw_torch").
        seed=42,                                                     # Seed per la riproducibilità degli esperimenti.
        data_seed=42,                                                # Seed per assicurare la riproducibilità durante lo shuffling dei dati.
    )

def create_trainer(model, training_args, train_dataset, eval_dataset, tokenizer):
    """Create and configure Trainer instance"""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model.config.use_cache = False
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

def train_model(trainer):
    """Execute training and evaluation"""
    log_gpu_info()
    trainer.train()
    return trainer.evaluate()

def save_model(trainer, output_dir):
    """Save model and tokenizer"""
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)