import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# === CONFIG ===
MODEL_NAME = "codellama/CodeLlama-7b-hf"
CSV_PATH = "mio_dataset.csv"  # <-- metti qui il tuo file CSV
CODE_FIELD = "code"           # <-- nome della colonna del codice
OUTPUT_DIR = "./codellama-7b-lora-csv"
MAX_LENGTH = 512

# === Quantization 4bit config ===
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# === Load tokenizer e modello quantizzato ===
print("Caricamento modello e tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# === Applica LoRA ===
print("Applicazione di LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# === Caricamento dataset ===
print("Caricamento dataset CSV...")
dataset = load_dataset("csv", data_files=CSV_PATH)["train"]

# === Tokenizzazione ===
print("Tokenizzazione dataset...")
def tokenize(example):
    return tokenizer(
        example[CODE_FIELD],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize)

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# === Training args ===
print("Inizio training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    learning_rate=2e-4,
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Training ===
trainer.train()

# === Save ===
print("Salvataggio finale...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completato!")
