import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc  # Importa le funzioni necessarie

def _ensure_dir(path):
    """Crea una directory se non esiste."""
    os.makedirs(path, exist_ok=True)

def _append_to_csv(data_dict, csv_path):
    """Aggiunge un dizionario come riga a un CSV, creando l'header se necessario."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

def save_plot(plot_name, output_dir, plt_object=None, subfolder="plots"):
    """
    Salva un plot in formato PNG.
    
    Args:
        plot_name (str): Nome del file (senza estensione).
        output_dir (str): Directory principale di output.
        plt_object (matplotlib.pyplot, optional): Oggetto figura. Se None, usa plt.gcf().
        subfolder (str): Sottocartella per i plot.
    """
    plot_dir = os.path.join(output_dir, subfolder)
    _ensure_dir(plot_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_name}_{timestamp}.png"
    filepath = os.path.join(plot_dir, filename)
    
    if plt_object is None:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt_object.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(plt_object)
    
    return filepath

def save_results(strategy, results, output_dir):
    """Salva i risultati finali per una strategia."""
    # Salva JSON
    file_path = os.path.join(output_dir, 'final_results.json')
    with open(file_path, 'w') as f:
        json.dump({
            'strategy': strategy,
            'results': results
        }, f, indent=4)
    # Salva CSV
    if isinstance(results, dict):
        csv_path = os.path.join(output_dir, 'csv', 'final_metrics.csv')
        _ensure_dir(os.path.dirname(csv_path))
        row_data = {'strategy': strategy, **results}
        _append_to_csv(row_data, csv_path)

def save_fold_results(strategy, fold_num, results, output_dir):
    """Salva i risultati per un singolo fold."""
    # Salva JSON
    json_path = os.path.join(output_dir, f'fold_{fold_num}_results.json')
    with open(json_path, 'w') as f:
        json.dump({'strategy': strategy, 'fold': fold_num, 'results': results}, f, indent=4)
    
    # Salva CSV
    if isinstance(results, dict):
        csv_path = os.path.join(output_dir, 'csv', 'fold_metrics.csv')
        _ensure_dir(os.path.dirname(csv_path))
        row_data = {'strategy': strategy, 'fold': fold_num, **results}
        _append_to_csv(row_data, csv_path)

def save_training_args(args, output_dir):
    """Salva gli argomenti di training per riferimento."""
    file_path = os.path.join(output_dir, 'training_args.json')
    with open(file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def save_roc_curve(y_true, y_probs, output_dir, plot_name="roc_curve"):
    """
    Salva la curva ROC come immagine e ritorna i valori AUC.
    
    Args:
        y_true (array): Etichette vere (binarie).
        y_probs (array): Probabilità predette per la classe positiva.
        output_dir (str): Directory di output.
        plot_name (str): Nome base del plot.
    
    Returns:
        dict: Dati della curva ROC e AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plot_path = save_plot(plot_name, output_dir)
    plt.close()
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc,
        'plot_path': plot_path
    }
