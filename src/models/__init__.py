"""Package `src.models` - implementazioni dei classificatori.

Il package separa due "famiglie" di modelli, scelte di proposito per
permettere un confronto baseline vs deep learning nel report finale:

- `classical.py` - SVM (RBF) e RandomForest su feature *handcrafted* (LBP +
  GLCM + HOG concatenate). Veloci, interpretabili, allenabili su CPU.

- `deep.py` - Fine-tuning a due fasi di backbone pre-addestrati
  (EfficientNet, DenseNet) tramite la libreria `timm`. Richiede una GPU
  CUDA (GPU NVIDIA locale - es. RTX 3080 - oppure T4 su Colab/Kaggle).

Entrambe le pipeline condividono lo stesso preprocessing (`src/preprocessing.py`)
e lo stesso schema di valutazione (`src/evaluate.py`) per permettere un
confronto equo.
"""
