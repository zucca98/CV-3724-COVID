"""Package `src` - codice sorgente principale del progetto.

Questo package contiene la libreria interna del progetto di classificazione
COVID/non-COVID su CT del torace. E' organizzato a moduli per responsabilita':

- `preprocessing.py` - I/O immagini, CLAHE, lung masking, augmentation.
- `features.py`      - Feature handcrafted (LBP, GLCM, HOG).
- `models/`          - Modelli classici (SVM, RF) e deep (EfficientNet, DenseNet).
- `postprocess.py`   - Calibrazione, threshold tuning, TTA, ensemble, Grad-CAM.
- `evaluate.py`      - Metriche e plot per la valutazione.
- `utils.py`         - Funzioni trasversali: seed, logging, config loading.

I file in questo package NON contengono entry point CLI: per quelli vedi
`scripts/`. Qui ci sono solo funzioni/classi importabili.
"""
