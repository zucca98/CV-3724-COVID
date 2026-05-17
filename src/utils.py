"""Funzioni di utilita' trasversali al progetto.

Scopo del file
--------------
Questo modulo raccoglie tre helper "infrastrutturali" che vengono usati ovunque
nel progetto e che hanno senso *fuori* dai moduli specifici (preprocessing,
features, modelli):

1. **set_seed**: fissa tutti i generatori di numeri casuali (Python, NumPy,
   PyTorch CPU e CUDA) ad uno stesso seed, in modo che esecuzioni successive
   producano gli stessi risultati. La riproducibilita' e' un vincolo esplicito
   del progetto (seed=42 ovunque).

2. **get_logger**: ritorna un logger preconfigurato con un formato uniforme,
   evitando di duplicare handler ad ogni nuovo import (controllo
   `if not logger.handlers`). Sostituisce le `print()`, vietate dentro
   `src/` per convenzione del progetto.

3. **load_config**: carica un file YAML (di default `configs/default.yaml`)
   in un dizionario Python. Usato dai vari script CLI per parametrizzare la
   pipeline senza hardcodare valori.

Dipendenze
----------
- `numpy`, `torch`: per la riproducibilita' dei rispettivi RNG.
- `yaml` (PyYAML): per il parsing dei file di configurazione.
"""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# pandas 3.0 abilita di default `future.infer_string`, che fa usare il backend
# PyArrow per le colonne stringa. Su Windows pyarrow va in access violation
# (segfault nativo) quando le sue DLL coesistono con quelle di torch: il crash
# e' riproducibile in qualsiasi `pd.read_csv` con colonne testuali. `utils` e'
# importato da ogni script/modulo, quindi disabilitando l'opzione qui il fix
# vale per l'intera pipeline prima di ogni lettura CSV.
try:
    pd.set_option("future.infer_string", False)
except (KeyError, ValueError):
    pass  # opzione assente su versioni di pandas precedenti alla 2.1


def set_seed(seed: int = 42) -> None:
    """Fissa i generatori di numeri casuali per garantire risultati riproducibili.

    Vengono inizializzati:
    - `random` (Python standard library)
    - `numpy.random`
    - PyTorch CPU (`torch.manual_seed`)
    - PyTorch CUDA su tutti i device disponibili (`cuda.manual_seed_all`)

    Nota: PyTorch puo' avere comunque kernel CUDA non-deterministici (es.
    cuDNN benchmark). Per riproducibilita' "stretta" servirebbe anche
    `torch.backends.cudnn.deterministic = True`, ma sacrificherebbe
    parecchia velocita': il progetto accetta micro-variazioni numeriche.

    Args:
        seed: Intero usato come seed comune. Default 42 (convenzione globale
            del progetto).

    Side effects:
        Modifica lo stato globale degli RNG di `random`, `numpy` e `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Anche se non c'e' una GPU, questa chiamata e' un no-op: e' sicura.
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Restituisce un logger di modulo configurato per scrivere su stderr.

    Pattern adottato per evitare la duplicazione degli handler quando lo
    stesso modulo viene importato piu' volte (cosa che produrrebbe log
    duplicati ad ogni `logger.info`): controlliamo `if not logger.handlers`
    prima di aggiungere uno StreamHandler.

    Formato dei messaggi:
        ``2024-01-15 10:30:01,123 INFO src.preprocessing: messaggio``

    Args:
        name: Nome del logger. Convenzionalmente `__name__` del modulo che lo
            richiede, in modo che la gerarchia dei logger rifletta la gerarchia
            dei package Python.

    Returns:
        Istanza `logging.Logger` pronta all'uso. Livello impostato a INFO.
    """
    logger = logging.getLogger(name)
    # Solo se il logger non e' gia' stato configurato: cosi' richiami
    # successivi non aggiungono handler duplicati.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Carica un file di configurazione YAML in un dizionario Python.

    Usa `yaml.safe_load` (non `yaml.load`) per evitare l'esecuzione di codice
    arbitrario incluso nel file YAML: e' la pratica corretta per file di
    configurazione non controllati direttamente.

    Args:
        path: Percorso del file YAML. Accetta sia `str` che `pathlib.Path`.
            Default: `configs/default.yaml`.

    Returns:
        Dizionario con la struttura del YAML (nested dict/list/scalar).

    Raises:
        FileNotFoundError: Se `path` non esiste.
        yaml.YAMLError: Se il file non e' un YAML valido.
    """
    with open(path) as f:
        return yaml.safe_load(f)
