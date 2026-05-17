"""Modello deep learning: backbone timm con fine-tuning a due fasi.

Scopo del file
--------------
Questo modulo implementa il "braccio deep" della pipeline. Usa la libreria
`timm` (PyTorch Image Models) per caricare backbone pre-addestrati su
ImageNet (EfficientNet-B0 per default, ma supporta anche DenseNet121 ecc.)
e li specializza al task di classificazione COVID/non-COVID tramite
fine-tuning a due fasi.

Strategia di fine-tuning "a due fasi"
-------------------------------------
Questa e' la strategia standard per fare *transfer learning* su dataset
piccoli, evitando di distruggere i pesi pre-addestrati:

**Fase 1 - Solo testa**:
    - Congela TUTTO il backbone (`requires_grad = False`).
    - Allena SOLO la nuova testa lineare di classificazione.
    - LR alto (es. 1e-3): la testa parte da zero, puo' permetterselo.
    - Obiettivo: portare la testa a fare predizioni "ragionevoli" senza
      perturbare le feature pre-apprese.

**Fase 2 - Sblocco parziale**:
    - Ricarica i pesi migliori della Fase 1.
    - Sblocca solo gli ultimi 2 blocchi del backbone + conv_head + bn2.
      Idea: i layer "alti" del backbone catturano feature task-specific,
      quelli "bassi" sono generici (bordi, texture) e vanno preservati.
    - LR molto piu' basso (es. 1e-5): aggiustamenti fini ai pesi
      pre-addestrati, NON ri-allenamento da zero.
    - Weight decay attivo per regolarizzazione.
    - Early stopping sulla val_loss per evitare overfitting.

Per entrambe le fasi: AMP (mixed precision) se siamo su CUDA, scheduler
`ReduceLROnPlateau` che dimezza il learning rate quando la val_loss
stagna, log per-epoch su CSV.

Componenti principali
---------------------
- `CovidCTModel`     - `nn.Module` con backbone + testa lineare.
- `build_model`      - factory che assembla il modello tramite timm.
- `CovidCTDataset`   - `torch.utils.data.Dataset` che applica al volo la
                       pipeline di preprocessing (resize, lung mask, CLAHE).
- `train_one_epoch`  - loop di training di una epoca (con AMP opzionale).
- `validate_one_epoch` - loop di validation con calcolo di ROC-AUC.
- `_unfreeze_last_blocks` - sblocco selettivo degli ultimi N stage del
                             backbone (gestisce sia EfficientNet sia
                             architetture senza `.blocks`).
- `fit`              - orchestra le due fasi di training end-to-end.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd

from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    load_image,
    lung_mask,
    resize_with_padding,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Mappa stringa -> intero per le label. COVID = classe positiva (1).
# La scelta di indicare COVID come positivo NON e' arbitraria: tutte le
# metriche tipo "recall" / "precision" sono calcolate rispetto alla classe
# positiva, e nel nostro task vogliamo monitorare quanto il modello prende
# bene i COVID (sensibilita').
_LABEL_MAP: dict[str, int] = {"covid": 1, "non-covid": 0}


# =============================================================================
# Architettura del modello
# =============================================================================

class CovidCTModel(nn.Module):
    """Backbone EfficientNet/DenseNet con testa di classificazione custom.

    L'idea: prendiamo un backbone pre-addestrato su ImageNet (1000 classi),
    rimuoviamo la sua testa originale e ne attacchiamo una nostra a 2 classi.
    Dropout prima della linear layer per regolarizzazione.

    Args:
        backbone: Modulo timm creato con `num_classes=0` (testa rimossa).
            Per esempio: `timm.create_model('efficientnet_b0', num_classes=0)`.
        in_features: Dimensione di output del backbone (= `backbone.num_features`).
        num_classes: Numero di classi in output. Default 2 (binario).
        dropout: Probabilita' di dropout prima della linear. Default 0.5
            (forte regolarizzazione, sensata per dataset piccoli).
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # `nn.Sequential` per impacchettare Dropout + Linear in un unico modulo
        # piu' facile da congelare/sbloccare separatamente dal backbone.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward: backbone -> feature vector -> head -> logits (no softmax!).
        # Il softmax viene applicato altrove (nelle metriche / postprocess).
        return self.head(self.backbone(x))


def build_model(
    arch: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> CovidCTModel:
    """Crea un'istanza `CovidCTModel` con backbone caricato da timm.

    Args:
        arch: Nome dell'architettura timm. Esempi: `efficientnet_b0`,
            `efficientnet_b3`, `densenet121`. Vedi `timm.list_models()` per
            l'elenco completo.
        num_classes: Numero di classi in output.
        pretrained: Se True, carica i pesi pre-addestrati su ImageNet.
            Default True (transfer learning).
        dropout: Probabilita' dropout nella testa.

    Returns:
        Istanza `CovidCTModel` pronta per il training.
    """
    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        # `num_classes=0` dice a timm: "ritorna solo le feature, NON la testa
        # di classificazione". Cosi' possiamo attaccarne una nostra senza
        # dover "tagliare" manualmente l'ultimo layer.
        num_classes=0,
        # Mantenuto il global average pooling: trasforma le feature map 2D
        # finali in un vettore feature 1D (dimensione `num_features`).
        global_pool="avg",
    )
    in_features: int = backbone.num_features
    logger.info("Built %s  in_features=%d  pretrained=%s", arch, in_features, pretrained)
    return CovidCTModel(backbone, in_features, num_classes, dropout)


# =============================================================================
# Dataset PyTorch
# =============================================================================

class CovidCTDataset(Dataset):
    """Dataset CT che applica la pipeline di preprocessing al volo (per ogni `__getitem__`).

    Filosofia: NON pre-calcoliamo gli embedding o le immagini preprocessate
    su disco. Ogni volta che il DataLoader chiede un sample, applichiamo
    l'intera catena (load -> grayscale -> resize -> lung mask -> 3-channel ->
    augmentation). Questo costa CPU ma:
    - Permette augmentation diverse ad ogni epoca.
    - Evita di occupare GB di spazio disco con tensori pre-calcolati.
    - Rende il codice piu' semplice e meno error-prone.

    Per CPU lenti, l'opzione `apply_mask=False` salta la lungmask (la parte
    piu' costosa) - utile per smoke test rapidi.

    Args:
        df: DataFrame con colonne `filepath` (path al PNG) e `label`
            (`'covid'` o `'non-covid'`).
        transform: Compose albumentations - deve includere `Normalize` +
            `ToTensorV2`.
        apply_mask: Se applicare la lung mask prima di costruire
            l'immagine a 3 canali. Default True. Disabilitalo per smoke test.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        apply_mask: bool = True,
    ) -> None:
        # `reset_index(drop=True)` garantisce che `df.iloc[idx]` funzioni con
        # indici 0..len-1 anche se il DataFrame originale aveva indici sparsi
        # (es. risultato di un filtro pandas).
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.apply_mask = apply_mask

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Restituisce la coppia `(tensor, label)` per il sample `idx`.

        Pipeline applicata:
            load PNG -> grayscale -> resize 224 -> [lung mask] -> 3-channel ->
            transform (normalize + ToTensor) -> tensore `(3, 224, 224)`.
        """
        row = self.df.iloc[idx]
        img = load_image(Path(row["filepath"]))

        # Garantiamo che l'immagine sia in scala di grigi (alcune immagini
        # del dataset originale sono RGB anche se concettualmente sono CT).
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = resize_with_padding(gray, target=224)

        if self.apply_mask:
            mask = lung_mask(gray)
            gray = apply_lung_mask(gray, mask)

        # Costruiamo l'immagine "RGB sintetica" a 3 canali (grayscale + 2
        # CLAHE) - vedi `build_three_channel` per dettagli.
        img3 = build_three_channel(gray)                  # (224, 224, 3) uint8
        # `transform` applica augmentation (in training) + normalize +
        # ToTensorV2 (converte HWC uint8 in CHW float32 normalizzato).
        tensor = self.transform(image=img3)["image"]      # (3, 224, 224) float32

        label = _LABEL_MAP[row["label"]]
        return tensor, label


# =============================================================================
# Loop di training/validation per una singola epoca
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Esegue una epoca di training.

    Supporta Automatic Mixed Precision (AMP): se `scaler` non e' `None`,
    le moltiplicazioni nei layer compatibili avvengono in float16 (piu'
    veloci e meno memoria su GPU moderne) con loss in float32 scalata per
    evitare underflow del gradiente.

    Args:
        model: Modello in training mode (verra' messo in train mode).
        loader: DataLoader del training split.
        optimizer: Ottimizzatore (Adam in fase 1, AdamW in fase 2).
        criterion: Loss function (tipicamente CrossEntropyLoss).
        device: Dispositivo di calcolo (CPU o CUDA).
        scaler: `torch.amp.GradScaler` per AMP. `None` disabilita AMP.

    Returns:
        Dizionario con `loss` (media pesata sull'intero set) e `accuracy`.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = scaler is not None

    # `leave=False` fa sparire la barra tqdm a fine epoca, mantenendo pulito
    # l'output (i log per-epoca vengono stampati separatamente).
    for X, y in tqdm(loader, desc="  train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Autocast: dentro questo context manager le operazioni vengono
        # eseguite in float16 dove possibile (matmul, conv). Se AMP e'
        # disabilitato, e' un no-op.
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(X)
            loss = criterion(logits, y)

        if amp_enabled:
            # AMP path: scalare la loss prima del backward per evitare che
            # gradienti molto piccoli vadano a zero in float16.
            scaler.scale(loss).backward()
            scaler.step(optimizer)   # internamente "unscale" + optimizer.step
            scaler.update()          # adatta la scala in base agli inf/NaN visti
        else:
            loss.backward()
            optimizer.step()

        n = len(y)
        # Moltiplichiamo per `n` perche' la loss e' gia' una media sul batch:
        # vogliamo la media pesata corretta a fine epoca.
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += n

    return {"loss": total_loss / total, "accuracy": correct / total}


def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Esegue una epoca di validation.

    Calcola loss, accuracy e ROC-AUC. Importante: NON usa AMP nemmeno se
    abilitato in training - vogliamo metriche numericamente precise (la
    differenza float16 vs float32 e' tipicamente trascurabile ma meglio
    evitare di introdurre rumore qui).

    Args:
        model: Modello (verra' messo in eval mode).
        loader: DataLoader del validation split.
        criterion: Loss function.
        device: Dispositivo di calcolo.

    Returns:
        Dizionario con `loss`, `accuracy`, `roc_auc`.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    # `no_grad`: niente computational graph -> meno memoria, piu' velocita'.
    with torch.no_grad():
        for X, y in tqdm(loader, desc="  val  ", leave=False):
            X, y = X.to(device), y.to(device)

            # Autocast disabilitato esplicitamente: precisione full FP32 per
            # i numeri usati nelle metriche.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                logits = model(X)
                loss = criterion(logits, y)

            n = len(y)
            total_loss += loss.item() * n
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += n

            # Raccogliamo prob e label per il calcolo cumulativo dell'AUC a
            # fine epoca (AUC NON e' calcolabile incrementalmente).
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    # ROC-AUC richiede entrambe le classi nel set; se per qualche motivo
    # nello split di validation appare una sola classe, ritorniamo 0
    # (segnale di anomalia, non crash).
    auc = (
        roc_auc_score(all_labels, all_probs)
        if len(set(all_labels)) > 1
        else 0.0
    )
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "roc_auc": float(auc),
    }


# =============================================================================
# Two-phase fine-tuning
# =============================================================================

def _unfreeze_last_blocks(backbone: nn.Module, n: int = 2) -> None:
    """Sblocca gli ultimi `n` stage del backbone + i layer "neck" finali.

    Strategia adattiva: la maggior parte dei backbone moderni (EfficientNet,
    MobileNet) ha un attributo `backbone.blocks` che contiene gli stage
    convoluzionali. Per altre architetture (DenseNet, ResNet) la struttura
    e' diversa: in quel caso facciamo fallback sui "top-level children" con
    parametri.

    Layer "neck" di EfficientNet (`conv_head`, `bn2`) e DenseNet (`norm`,
    `head_norm`) vengono sempre sbloccati se presenti - sono i layer subito
    prima della classification head e beneficiano del fine-tuning.

    Args:
        backbone: Il modulo backbone (es. `model.backbone`).
        n: Numero di stage finali da sbloccare. Default 2.

    Side effects:
        Modifica `requires_grad=True` per i parametri selezionati.
    """
    blocks_container = getattr(backbone, "blocks", None)
    if blocks_container is not None:
        # Caso EfficientNet/MobileNet: `blocks` e' un container ordinato di
        # stage convoluzionali.
        stages = list(blocks_container.children())
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        # Layer "neck" subito dopo i blocks: sbloccali se esistono.
        # `getattr(...,  None)` evita AttributeError per architetture senza
        # questi attributi.
        for attr in ("conv_head", "bn2", "norm", "head_norm"):
            mod = getattr(backbone, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
    else:
        # Fallback per architetture non-EfficientNet: prendi gli ultimi N
        # children top-level che hanno parametri (saltando ad es. ReLU senza
        # weight).
        children_with_params = [
            c for _, c in backbone.named_children() if list(c.parameters())
        ]
        for child in children_with_params[-n:]:
            for p in child.parameters():
                p.requires_grad = True


def fit(
    model: CovidCTModel,
    train_loader,
    val_loader,
    cfg: dict,
    output_dir: Path,
) -> Path:
    """Orchestratore del workflow di fine-tuning a due fasi.

    Comportamento dettagliato:

    Fase 1 - Solo testa:
        - `backbone.parameters().requires_grad = False`.
        - Ottimizzatore: `Adam(head.parameters(), lr=lr_phase1)`.
        - Per `epochs_phase1` epoche: train + validate + scheduler step.
        - Best checkpoint salvato (val_loss minima).

    Fase 2 - Sblocco parziale:
        - Ricarica il best checkpoint Fase 1.
        - Sblocca gli ultimi 2 stage del backbone + neck.
        - Ottimizzatore: `AdamW(trainable params, lr=lr_phase2, wd=weight_decay)`.
        - Per fino a `epochs_phase2` epoche, con early stopping
          (`early_stopping_patience` epoche senza miglioramento -> stop).

    Logging:
        - Log CSV per-epoca in `output_dir/training_log.csv`.
        - Best weights in `output_dir/best_model.pth`.

    Args:
        model: Istanza di `CovidCTModel`.
        train_loader: DataLoader training.
        val_loader: DataLoader validation.
        cfg: Dizionario configurazione. Chiavi attese:
            `epochs_phase1`, `epochs_phase2`, `lr_phase1`, `lr_phase2`,
            `weight_decay`, `early_stopping_patience`, `use_amp`.
        output_dir: Directory dove salvare checkpoint e log.

    Returns:
        Path del file `.pth` con i pesi del modello migliore.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    log_path = output_dir / "training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = model.to(device)

    # AMP attivo solo su CUDA: su CPU non c'e' guadagno (anzi, puo' rallentare).
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    criterion = nn.CrossEntropyLoss()

    # Header CSV: scritto una sola volta all'inizio.
    _log_header = [
        "epoch", "phase", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_auc", "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(_log_header)

    def _append_log(row: list) -> None:
        """Append helper - apre il CSV in modalita' append e scrive una riga."""
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # -------------------------------------------------- Phase 1: solo head ----
    logger.info("=== Phase 1: head-only training (%d epochs) ===", cfg["epochs_phase1"])
    # Congeliamo TUTTO il backbone: solo la testa avra' gradienti.
    for p in model.backbone.parameters():
        p.requires_grad = False

    # In fase 1 alleniamo solo i parametri della testa - LR alto perche'
    # partono da inizializzazione casuale.
    optimizer = Adam(model.head.parameters(), lr=cfg["lr_phase1"])
    # ReduceLROnPlateau: se val_loss non migliora per `patience` epoche,
    # dimezza il LR (factor=0.5).
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, cfg["epochs_phase1"] + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        # Estraiamo il LR corrente (potrebbe essere stato modificato dallo
        # scheduler all'epoca precedente).
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["loss"])

        logger.info(
            "P1 ep%02d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e",
            epoch, train_m["loss"], val_m["loss"], val_m["roc_auc"], lr_now,
        )
        _append_log([epoch, 1, train_m["loss"], train_m["accuracy"],
                     val_m["loss"], val_m["accuracy"], val_m["roc_auc"], lr_now])

        # Salvataggio del miglior modello (per val_loss). Usiamo val_loss
        # invece di accuracy perche' e' un segnale piu' "liscio" e meno
        # rumoroso su classi sbilanciate.
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(model.state_dict(), best_path)

    # ----------------------------- Phase 2: sblocco parziale + fine-tuning ----
    logger.info("=== Phase 2: fine-tuning last 2 blocks (%d epochs, patience=%d) ===",
                cfg["epochs_phase2"], cfg["early_stopping_patience"])
    # Ricarichiamo i pesi MIGLIORI di Fase 1 (non quelli dell'ultima epoca!).
    model.load_state_dict(torch.load(best_path, map_location=device))
    _unfreeze_last_blocks(model.backbone, n=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters after unfreeze: %d", trainable)

    # AdamW invece di Adam: stessa logica ma con weight decay applicato
    # *correttamente* (separato dalla update step di Adam). Best practice
    # per fine-tuning.
    # Filtriamo i parametri con `requires_grad=True` per non passare i
    # frozen all'ottimizzatore (otterremmo uno spreco di memoria).
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr_phase2"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    no_improve = 0   # contatore per early stopping

    for epoch in range(1, cfg["epochs_phase2"] + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["loss"])

        logger.info(
            "P2 ep%02d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e",
            epoch, train_m["loss"], val_m["loss"], val_m["roc_auc"], lr_now,
        )
        _append_log([epoch, 2, train_m["loss"], train_m["accuracy"],
                     val_m["loss"], val_m["accuracy"], val_m["roc_auc"], lr_now])

        # Salvataggio + reset del contatore di "no improvement".
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping: se per N epoche consecutive val_loss non migliora,
        # interrompiamo per evitare overfitting e risparmiare tempo.
        if no_improve >= cfg["early_stopping_patience"]:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, cfg["early_stopping_patience"])
            break

    logger.info("Best val_loss: %.4f  -> %s", best_val_loss, best_path)
    return best_path
