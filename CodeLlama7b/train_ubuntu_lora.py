from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 1. Parametri
MODEL_NAME = "codellama/CodeLlama-7b-hf"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./codellama-lora-output"

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Evita errori se manca <pad>

# 3. Carica modello base
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,  # Usa quantizzazione 8bit con bitsandbytes
    device_map="auto"
)

# 4. Applica configurazione LoRA
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# 5. Dataset (puoi sostituire con uno tuo in italiano)
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")  # Mini subset per esempio

# 6. Preprocessing
def tokenize(example):
    prompt = f"### Istruzione:\n{example['instruction']}\n### Input:\n{example['input']}\n### Risposta:\n{example['output']}"
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 10. Addestramento
trainer.train()

# 11. Salvataggio adattatore LoRA
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)