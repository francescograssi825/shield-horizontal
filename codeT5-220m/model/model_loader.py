from transformers import AutoTokenizer, T5ForSequenceClassification, AutoModelForSequenceClassification

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, config):
    tokenizer = load_tokenizer(model_name)
    if "codebert" in model_name.lower():
        # Load the CodeBERT model natively using AutoModelForSequenceClassification.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=config['id2label'],
            label2id=config['label2id'],
            ignore_mismatched_sizes=True
        )
    else:
        # Default to loading a CodeT5 model.
        model = T5ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=config['id2label'],
            label2id=config['label2id']
        )
        
    # Set the model configuration's pad token id using the tokenizer's pad token.
    model.config.pad_token_id = tokenizer.pad_token_id
    return model



from transformers import AutoTokenizer, AutoModelForSequenceClassification

