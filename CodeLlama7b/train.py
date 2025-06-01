from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Step 1: Configurazione
MODEL_NAME = "codellama/CodeLlama-7b-hf"
OUTPUT_DIR = "./codellama-7b-lora"
USE_4BIT = True
MAX_LENGTH = 512

# Step 2: Carica tokenizer e modello
print("🔄 Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)


# Step 3: Configura LoRA
print("⚙️ Configurazione LoRA...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# Step 4: Carica dataset di esempio
print("📦 Caricamento dataset...")
dataset = load_dataset("code_search_net", split="train[:1%]")  # sottoinsieme

# Step 5: Tokenizzazione
print("✏️ Tokenizzazione...")
def tokenize_function(example):
    return tokenizer(
        example["code"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = load_dataset("csv", data_files="mio_dataset.csv")["train"]

def tokenize(example):
    return tokenizer(
        example["code"],  # cambia con il nome della colonna
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize)


# Step 6: Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 7: Parametri di training
print("🚀 Inizio training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,
    learning_rate=2e-4,
    report_to="none"
)

# Step 8: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 9: Allenamento
trainer.train()

# Step 10: Salvataggio finale
print("💾 Salvataggio modello...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning completato.")
