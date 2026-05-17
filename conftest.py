"""Configurazione globale di pytest per il progetto.

Scopo del file
--------------
`conftest.py` viene **caricato automaticamente da pytest** prima di
qualsiasi test (e' una convenzione del framework, NON va importato
esplicitamente). Lo usiamo per:

1. **Forzare il backend non-interattivo di matplotlib** (`Agg`): nei test
   creiamo grafici, ma in CI/ambiente headless non esiste un display X11
   (Tk fallirebbe con `ImportError`). `Agg` rende il rendering puro CPU
   -> file PNG, senza GUI.

2. **Aggiungere la root del progetto a `sys.path`**: cosi' i test possono
   fare `from src.preprocessing import ...` senza dover prima
   `pip install -e .` ne' settare `PYTHONPATH=.` a mano.

Convenzione pytest
------------------
Un `conftest.py` posizionato qui (root del progetto) viene applicato a
TUTTI i test sottostanti. Si possono aggiungere altri `conftest.py` in
sottocartelle per scope locali (`tests/integration/conftest.py`, ecc.) -
non necessario per ora.
"""
import sys
from pathlib import Path

import matplotlib
# `use("Agg")` DEVE essere chiamato PRIMA di qualsiasi `import matplotlib.pyplot`.
# Mettendolo qui ci assicuriamo che sia applicato globalmente, anche se
# qualche modulo importato dai test prova a usare un backend GUI.
matplotlib.use("Agg")

# Aggiunge la root del progetto al `sys.path` cosi' che `import src.*`
# funzioni dai test senza configurare PYTHONPATH manualmente. `Path(__file__)`
# punta a questo file (`conftest.py`); `.parent` -> directory del progetto.
sys.path.insert(0, str(Path(__file__).parent))
