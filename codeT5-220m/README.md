# shield-horizontal

# CodeT5 Binary Classifier


## Installazione
```bash
pip install -r requirements.txt

python run.py --strategy [standard|kfold|stratified_kfold] --k [num_folds] --model Salesforce/codet5-small
Parametri disponibili:

--strategy: Strategia di validazione (default: standard)

--k: Numero di fold per le strategie kfold (default: 5)

--model: Nome del modello HuggingFace (default: Salesforce/codet5-small)

--epochs: Numero di epoche (default: 3)

--batch_size: Dimensione del batch (default: 8)

--max_length: Lunghezza massima sequenza (default: 256)

--data_path: Percorso al dataset (default: datasets/training_dataset.csv)


# Esecuzione standard
python run.py --strategy standard


python run.py --strategy standard --model huggingface/CodeBERTa-language-id

python run.py --strategy standard --model huggingface/CodeBERTa-language-id --max_length 512


# K-Fold con 10 fold
python run.py --strategy kfold --k 10

# Stratified K-Fold con modello diverso
python run.py --strategy stratified_kfold --model Salesforce/codet5-base
