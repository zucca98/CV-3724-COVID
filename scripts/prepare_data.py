"""Crea gli split train/val/test in formato CSV dal dataset SARS-CoV-2 CT-Scan.

Scopo dello script
------------------
E' il PRIMO passo della pipeline. Scansiona la cartella del dataset (con
sottocartelle `COVID/` e `non-COVID/` contenenti PNG), elenca tutti i file
e produce tre file CSV:

    data/processed/train.csv   (~70% del dataset, stratificato)
    data/processed/val.csv     (~15% del dataset, stratificato)
    data/processed/test.csv    (~15% del dataset, stratificato)

Ogni CSV ha tre colonne: `filepath`, `label`, `split`.

Vincoli di dominio rispettati
-----------------------------
- **Split stratificato** sulla label, per mantenere la stessa prevalenza di
  COVID/non-COVID nei tre split (importante per i dataset sbilanciati).
- **Seed fisso** (default 42, da `configs/default.yaml`) per riproducibilita':
  due esecuzioni successive producono gli stessi split.
- **Nessun patient ID disponibile** nel dataset: questo e' un limite noto
  che rappresenta un potenziale data leakage e va dichiarato
  esplicitamente nel report.

Uso
---
    PYTHONPATH=. python scripts/prepare_data.py
    PYTHONPATH=. python scripts/prepare_data.py --data-dir custom/path
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)

# Mapping da nome cartella (case sensitive del dataset originale) a label
# usata internamente (lowercase, con trattino). Le label nei CSV saranno
# `covid` / `non-covid`.
_LABEL_MAP = {
    "COVID": "covid",
    "non-COVID": "non-covid",
}


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    """Scansiona `data_dir/COVID/` e `data_dir/non-COVID/` e ritorna un DataFrame.

    Per ogni PNG trovato, registra path, label e dimensioni originali
    (utili per EDA e debugging). Le dimensioni vengono lette dall'header
    PNG (operazione velocissima, NON carica i pixel).

    Args:
        data_dir: Directory radice del dataset. Deve contenere le
            sottocartelle `COVID/` e `non-COVID/`.

    Returns:
        DataFrame con colonne `filepath` (str assoluto), `label`,
        `original_height`, `original_width`.

    Raises:
        FileNotFoundError: Se una delle due sottocartelle attese non esiste.
    """
    records: list[dict] = []
    for folder, label in _LABEL_MAP.items():
        folder_path = data_dir / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Expected subfolder not found: {folder_path}")
        # `sorted(...)` garantisce un ordine deterministico (importante per
        # la riproducibilita': anche `Path.glob` non garantisce ordine).
        png_files = sorted(folder_path.glob("*.png"))
        for p in tqdm(png_files, desc=f"Scanning {folder}", leave=False):
            # `with Image.open(...)` legge solo l'header PNG (rapidissimo)
            # e poi chiude correttamente il file handle.
            with Image.open(p) as img:
                w, h = img.size
            records.append(
                {
                    "filepath": str(p),
                    "label": label,
                    "original_height": h,
                    "original_width": w,
                }
            )
    return pd.DataFrame(records)


def make_splits(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stratificato 70/15/15 (train/val/test).

    Strategia in due passi (`train_test_split` non supporta direttamente
    split a 3 vie):
        1. Split iniziale 70/30 -> (train, temp).
        2. Split del 30% rimanente in 50/50 -> (val, test).
        Risultato finale: 70/15/15.

    Args:
        df: DataFrame con almeno la colonna `label`.
        seed: Seed per il random shuffler. Default 42.

    Returns:
        Tripletta `(train_df, val_df, test_df)`, ciascuno con una nuova
        colonna `split` con valore `'train'/'val'/'test'`.
    """
    # `stratify=df["label"]` garantisce che la proporzione di COVID/non-COVID
    # sia preservata in entrambe le parti dello split.
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    # Secondo split sul 30% temp: 50/50 -> val=15% + test=15%.
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["label"]
    )
    # `.copy()` evita warning di SettingWithCopyWarning quando assegniamo
    # la nuova colonna a una "view" del DataFrame.
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return train, val, test


def _log_split_stats(name: str, df: pd.DataFrame) -> None:
    """Logga il numero di campioni per classe in uno split.

    Side effects:
        Scrive su logger una riga con totale + breakdown per classe.
    """
    counts = df["label"].value_counts()
    logger.info(
        "%s - total: %d  |  covid: %d  |  non-covid: %d",
        name,
        len(df),
        counts.get("covid", 0),
        counts.get("non-covid", 0),
    )


def main() -> None:
    """Entry point CLI.

    Workflow:
        1. Parsing argomenti + caricamento config.
        2. Fix del seed.
        3. Scan del dataset -> DataFrame.
        4. Split stratificato 70/15/15.
        5. Salvataggio dei tre CSV in `out_dir/`.
        6. Log delle statistiche per split.

    Side effects:
        Scrive `train.csv`, `val.csv`, `test.csv` nella directory di output.
    """
    parser = argparse.ArgumentParser(description="Prepare stratified dataset splits.")
    # Tutti i path sono `Path` per uso uniforme con `pathlib`.
    # `default=None` indica "usa quello del config file".
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    # CLI override: l'argomento --data-dir, se passato, vince sul config.
    data_dir = args.data_dir or Path(data_cfg["raw_dir"])
    out_dir = args.out_dir or Path(data_cfg["processed_dir"])
    seed = data_cfg["seed"]

    set_seed(seed)
    logger.info("Scanning dataset at %s", data_dir)
    df = scan_dataset(data_dir)
    logger.info("Total images found: %d", len(df))

    train, val, test = make_splits(df, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Salviamo solo le colonne necessarie per le fasi successive
    # (`original_height/width` sono utili solo per EDA, restano nel DataFrame
    # in memoria ma non finiscono nei CSV per pulizia).
    for split_df, split_name in [(train, "train"), (val, "val"), (test, "test")]:
        out_path = out_dir / f"{split_name}.csv"
        split_df[["filepath", "label", "split"]].to_csv(out_path, index=False)
        _log_split_stats(split_name, split_df)

    logger.info("Splits saved to %s", out_dir)


# Pattern standard: il modulo puo' essere importato (es. per i test) senza
# eseguire `main`. Lo si esegue solo se lanciato come script.
if __name__ == "__main__":
    main()
