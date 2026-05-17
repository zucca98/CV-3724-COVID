"""Utility di post-processing: calibrazione, tuning della soglia, TTA, ensemble, Grad-CAM.

Scopo del file
--------------
Questo modulo contiene tutti gli step *dopo* che il modello ha emesso i suoi
logit/probabilita' grezzi. Sono tecniche standard per migliorare l'usabilita'
clinica delle predizioni di una rete neurale:

1. **Calibrazione (temperature scaling)** - Una rete neurale tipicamente e'
   "overconfident": dice "sono sicuro al 99%" anche quando ha solo l'80% di
   accuratezza. Temperature scaling divide i logit per uno scalare T appreso
   sul validation set, "ammorbidendo" la distribuzione softmax e rendendo le
   probabilita' interpretabili come *frequenze* reali. E' un metodo
   *post-hoc*: si applica al modello gia' allenato, senza ri-training.

2. **Threshold tuning per sensibilita'** - In ambito medico la `sensibilita'`
   (= recall sui malati) e' piu' critica della `specificita'` (= recall sui
   sani): un falso negativo (paziente COVID mancato) e' molto piu' grave di
   un falso positivo. Tariamo dunque la soglia di decisione affinche'
   sensibilita' >= 95% sul validation set, anche a costo di abbassare la
   specificita'.

3. **TTA (Test-Time Augmentation)** - Si applicano k augmentation (flip,
   rotazioni, ecc.) sull'immagine di test, si calcola la predizione di
   ciascuna versione e si media. Tecnica gratuita per ridurre la varianza
   delle predizioni.

4. **Ensemble** - Media pesata delle predizioni di piu' modelli (es.
   EfficientNet + DenseNet) per ottenere robustezza contro errori
   sistematici di un singolo modello.

5. **Grad-CAM** - Mappa di salienza che evidenzia "dove guarda" la rete
   quando classifica una particolare classe. Cruciale per interpretabilita'
   e per costruire fiducia clinica.

Integrazione
------------
- Usato da `scripts/calibrate_and_evaluate.py` e
  `scripts/run_full_evaluation.py` (pipeline finale di valutazione).
- Dipende da `src/preprocessing.get_eval_transform` per applicare la stessa
  normalizzazione del training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score

from src.preprocessing import get_eval_transform
from src.utils import get_logger

logger = get_logger(__name__)


def temperature_scaling(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Trova la temperatura ottimale T minimizzando la NLL sul validation set.

    Algoritmo (Guo et al., 2017 - "On Calibration of Modern Neural Networks"):
    - Parametrizza T come scalare positivo, inizializzato a 1.5 (valori > 1
      tipicamente sono il punto di partenza ragionevole per reti
      overconfident).
    - Minimizza `CrossEntropyLoss(logits / T, labels)` rispetto a T (i logit
      e le label sono fissi).
    - Usa L-BFGS: ottimizzatore quasi-Newton, ideale quando il problema e'
      basso-dimensionale (qui un solo parametro!) e differenziabile.
    - `T.clamp(min=1e-4)` evita divisione per zero/T negativo in caso di
      step dell'ottimizzatore troppo aggressivo.

    La NLL e' una *proper scoring rule*: minimizzarla calibra effettivamente
    le probabilita' (al contrario, minimizzare l'accuracy non lo farebbe).

    Args:
        logits_val: Logit grezzi del modello su validation, shape `(N, 2)`.
        y_val: Label intere `{0, 1}`, shape `(N,)`.

    Returns:
        Scalare `T > 0`:
        - `T > 1` ammorbidisce la softmax (modello overconfident, caso tipico).
        - `T < 1` la rende piu' "piccata" (modello sotto-confident, raro).

    Side effects:
        Logga il valore finale di T sul logger di modulo.
    """
    logits_t = torch.tensor(logits_val, dtype=torch.float32)
    labels_t = torch.tensor(y_val, dtype=torch.long)

    # Parametro learnable inizializzato a 1.5 (valore tipico iniziale).
    T = nn.Parameter(torch.ones(1) * 1.5)
    # L-BFGS richiede una `closure` che ricalcoli loss + backward ad ogni
    # iterazione interna (a differenza degli ottimizzatori first-order).
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        # `clamp(min=1e-4)` protegge da T -> 0 (overflow nella divisione).
        loss = criterion(logits_t / T.clamp(min=1e-4), labels_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    T_val = max(float(T.item()), 1e-4)
    logger.info("Temperature scaling: T = %.4f", T_val)
    return T_val


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Applica temperature scaling e restituisce le probabilita' calibrate della classe positiva.

    Conversione: `softmax(logits / T)`, poi prende la colonna 1 (= classe COVID).

    Args:
        logits: Logit grezzi `(N, 2)`.
        T: Temperatura (ottenuta da `temperature_scaling`).

    Returns:
        Vettore `(N,)` con probabilita' calibrate per la classe positiva.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits_t / T, dim=1).numpy()
    # Colonna 1 = probabilita' classe positiva (COVID, label=1).
    return probs[:, 1]


def tune_threshold_for_sensitivity(
    probs: np.ndarray,
    y_true: np.ndarray,
    target_sensitivity: float = 0.95,
) -> float:
    """Trova la soglia *piu' alta* che garantisce `sensibilita' >= target`.

    Strategia: ordino tutte le probabilita' uniche in ordine decrescente e le
    testo come candidati soglia, dalla piu' alta verso la piu' bassa. La
    prima che soddisfa il vincolo di sensibilita' e' quella ottima.

    Perche' "la piu' alta"? Perche' a parita' di sensibilita', una soglia
    piu' alta produce piu' specificita' (meno falsi positivi). Quindi
    cerchiamo la soglia *piu' restrittiva* che ancora rispetta il vincolo
    minimo di sensibilita'.

    Args:
        probs: Probabilita' della classe positiva, shape `(N,)`.
        y_true: Label vere `{0, 1}`, shape `(N,)`.
        target_sensitivity: Soglia minima di recall accettabile (default 0.95
            per allinearsi al vincolo del progetto: "Sensitivity > Specificity").

    Returns:
        Soglia di decisione in `[0, 1]`. Se nessuna soglia raggiunge il
        target (caso degenere), restituisce la probabilita' minima del
        dataset (-> predice tutto positivo, sensibilita' = 1).

    Side effects:
        Logga la soglia scelta e la sensibilita' ottenuta.
    """
    # Ordiniamo decrescente: vogliamo testare prima le soglie piu' "restrittive".
    candidates = np.sort(np.unique(probs))[::-1]
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        sens = recall_score(y_true, preds, zero_division=0)
        if sens >= target_sensitivity:
            logger.info(
                "Threshold %.4f  ->  sensitivity %.4f  (target %.2f)",
                thr, sens, target_sensitivity,
            )
            return float(thr)

    # Caso patologico: nessuna soglia raggiunge il target. Usiamo la prob
    # minima -> tutto classificato positivo -> sensibilita' = 1 ma
    # specificita' = 0 (avvisiamo l'utente con un warning).
    fallback = float(probs.min())
    logger.warning(
        "No threshold meets sensitivity %.2f; falling back to %.4f (predict all positive)",
        target_sensitivity, fallback,
    )
    return fallback


def tta_predict(
    model: nn.Module,
    image: np.ndarray,
    transforms_list: list,
    device: torch.device | None = None,
) -> np.ndarray:
    """Test-Time Augmentation: media le probabilita' softmax su piu' trasformazioni.

    Pipeline per ogni trasformazione in `transforms_list`:
        1. Applica la trasformazione all'immagine -> tensore `(C, H, W)`.
        2. Aggiunge la dimensione batch -> `(1, C, H, W)`.
        3. Forward del modello -> logit `(1, 2)`.
        4. Softmax -> probabilita' `(2,)`.
    Al termine, media le `K` probabilita' ottenute.

    Args:
        model: Modello allenato, gia' in eval mode (verra' comunque forzato).
        image: Immagine pre-processata `(H, W, 3)` uint8 (prima delle
            trasformazioni di normalizzazione).
        transforms_list: Lista di Compose albumentations (es. originale,
            flip orizzontale, +5deg, -5deg, ...). Ogni trasformazione deve
            includere `Normalize` + `ToTensorV2`.
        device: Dispositivo di calcolo. Se `None`, sceglie CUDA se disponibile.

    Returns:
        Vettore probabilita' mediato, shape `(2,)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_probs: list[np.ndarray] = []

    # `torch.no_grad()` disabilita il tracking dei gradienti: piu' veloce e
    # meno memoria durante l'inferenza.
    with torch.no_grad():
        for transform in transforms_list:
            # `unsqueeze(0)` aggiunge la dimensione batch (la rete si aspetta
            # un input 4D anche per una singola immagine).
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    return np.mean(all_probs, axis=0)


def ensemble_predict(
    models_weights: list[tuple[nn.Module, float]],
    image: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Ensemble pesato di piu' modelli su una singola immagine.

    Formula: `output = sum(w_i * softmax(model_i(x))) / sum(w_i)`.

    I pesi NON devono sommare a 1 (li normalizziamo noi): cosi' l'utente puo'
    passare pesi grezzi (es. accuratezza di validation) senza preoccuparsi
    della normalizzazione.

    Args:
        models_weights: Lista di coppie `(model, peso)`. Esempio:
            `[(effnet, 0.92), (densenet, 0.88)]`.
        image: Immagine pre-processata `(H, W, 3)` uint8.
        device: Dispositivo di calcolo. Auto-detect se `None`.

    Returns:
        Vettore probabilita' pesato, shape `(2,)`, dtype `float32`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tutti i modelli condividono la stessa pipeline di normalizzazione:
    # carichiamola una sola volta fuori dal loop.
    transform = get_eval_transform()
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    accumulated = np.zeros(2, dtype=np.float64)
    total_weight = 0.0

    for model, weight in models_weights:
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        accumulated += weight * probs
        total_weight += weight

    # Normalizzazione dei pesi: garantisce che le componenti dell'output
    # sommino a 1 (sono ancora probabilita').
    return (accumulated / total_weight).astype(np.float32)


def compute_gradcam(
    model: nn.Module,
    image: np.ndarray,
    target_layer: nn.Module,
    target_class: int = 1,
) -> np.ndarray:
    """Genera una heatmap Grad-CAM per l'immagine data.

    Grad-CAM (Selvaraju et al., 2017): per la classe target, calcola il
    gradiente del logit rispetto alle activation map dell'ultimo layer
    convoluzionale e usa la media globale di questi gradienti come "pesi"
    per combinare le activation map. Il risultato e' una mappa di salienza
    che mostra "dove guarda" il modello.

    Args:
        model: Modello allenato. Verra' forzato in eval mode dalla libreria.
        image: Immagine pre-processata `(H, W, 3)` uint8.
        target_layer: Modulo convoluzionale su cui agganciare gli hook.
            Per EfficientNet di solito `model.backbone.blocks[-1][-1]`.
        target_class: Indice della classe per cui calcolare la salienza.
            Default 1 = COVID.

    Returns:
        Heatmap `(H, W)` `float32`, valori in `[0, 1]` (1 = massima rilevanza).

    Raises:
        ImportError: Se `pytorch-grad-cam` non e' installato. Il pacchetto
            e' opzionale (richiesto solo per generare figure interpretative
            nel report), quindi facciamo import lazy.
    """
    # Import lazy: l'utente medio della pipeline non ha bisogno di Grad-CAM,
    # quindi non rendiamo questa libreria una dipendenza obbligatoria.
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        # `from exc` preserva l'eccezione originale nello stack trace,
        # utile per il debug.
        raise ImportError(
            "pytorch-grad-cam is required for compute_gradcam. "
            "Install it with: pip install grad-cam"
        ) from exc

    transform = get_eval_transform()
    input_tensor = transform(image=image)["image"].unsqueeze(0)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )
    # `grayscale_cam` ha shape (batch=1, H, W); restituiamo la singola
    # immagine 2D.
    return grayscale_cam[0]
