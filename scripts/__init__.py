"""Package `scripts` - entry point CLI del progetto.

Ogni modulo qui dentro espone una `main()` da lanciare via:

    PYTHONPATH=. python scripts/<nome>.py [opzioni]

Workflow tipico end-to-end (vedi anche `README.md`):

    1. prepare_data.py              -> data/processed/train|val|test.csv
    2. compute_norm_stats.py        -> configs/norm_stats.json
    3. extract_features.py          -> data/processed/features_*.npz
    4. train_classical.py           -> models/classical/{svm,rf}.pkl
    5. train_deep.py                -> models/deep/best_model.pth (richiede GPU)
    6. calibrate_and_evaluate.py    -> models/postprocess/calibration_results.json
    7. run_full_evaluation.py       -> docs/figures/*, logs/evaluation_results.json
    8. generate_qualitative_examples.py -> docs/figures/qualitative_examples.png

Tutti gli script accettano `--config configs/default.yaml` (default) e
fanno seed con `set_seed(cfg["data"]["seed"])` per riproducibilita'.
"""
