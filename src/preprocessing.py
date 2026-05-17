"""Utility di preprocessing per immagini CT del torace.

Scopo del file
--------------
Questo modulo raccoglie tutte le operazioni di **preparazione** delle immagini
TAC (Tomografia Computerizzata) del torace prima che vengano date in pasto sia
ai classificatori "classici" (SVM/RandomForest su feature handcrafted) sia ai
modelli "deep" (EfficientNet/DenseNet). Le funzioni qui contenute coprono:

1. **I/O immagini** (lettura PNG con OpenCV, conversione formati colore).
2. **Resize con padding** che preserva l'aspect ratio - evita di "schiacciare"
   il torace e mantiene proporzioni anatomiche corrette.
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation): tecnica
   classica per migliorare il contrasto locale nelle radiografie/CT senza
   amplificare il rumore.
4. **Lung mask** (segmentazione polmonare): isola la regione dei polmoni
   azzerando lo sfondo, in modo che il classificatore non apprenda artefatti
   "fuori dai polmoni" (per esempio scritte sui bordi delle immagini).
5. **Costruzione 3 canali** a partire da una singola slice grayscale - serve
   perche' i backbone pre-addestrati su ImageNet si aspettano un input a 3
   canali RGB.
6. **Pipeline di augmentation** (albumentations) per training/eval.

Vincoli di dominio importanti
-----------------------------
- **NIENTE FLIP VERTICALE**: anatomicamente errato per una CT del torace
  (cuore a sinistra, asimmetrie polmonari): un flip verticale produrrebbe
  immagini impossibili nel mondo reale.
- **Rotazioni limitate a +/- 10 gradi**: idem, rotazioni eccessive sono
  irrealistiche.
- **Normalizzazione con statistiche del training set**: mai usare medie/std
  calcolate su validation/test (data leakage). Le statistiche reali vengono
  calcolate da `scripts/compute_norm_stats.py` e salvate in
  `configs/norm_stats.json`. Se il file manca, si usano come fallback le
  statistiche di ImageNet (ragionevole proxy per il fine-tuning).

Integrazione con il resto del progetto
--------------------------------------
- E' importato da `src/features.py` (per estrarre feature handcrafted),
  da `src/models/deep.py` (Dataset PyTorch) e dagli script di training.
- Dipende da `src/utils.py` solo per il logger di progetto.
- Le funzioni sono *stateless* e *pure* (eccetto le augmentation che sono
  pseudo-random per definizione).
"""
import json
import math
from pathlib import Path

import cv2  # OpenCV: I/O immagini, CLAHE, morfologia, connectedComponents.
import numpy as np
import albumentations as A  # Libreria di augmentation per visione artificiale.
from albumentations.pytorch import ToTensorV2  # Converte ndarray HWC -> tensore CHW.

from src.utils import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Statistiche di normalizzazione
# -----------------------------------------------------------------------------
# Le medie/deviazioni standard per canale vengono calcolate *una sola volta*
# sul training split dallo script `scripts/compute_norm_stats.py` e
# serializzate in `configs/norm_stats.json`. Caricandole qui evitiamo di
# ricalcolarle ad ogni esecuzione (sono costanti, dipendono solo dal dataset).
#
# Se il file non esiste (es. su un clone appena fatto, prima di lanciare
# prepare_data + compute_norm_stats), facciamo fallback sulle statistiche
# canoniche di ImageNet: e' una scelta sensata quando si fa fine-tuning di un
# backbone pre-addestrato, perche' le statistiche del dataset sorgente sono
# comunque "vicine" a quelle di una CT normalizzata.
_NORM_STATS_PATH = Path(__file__).resolve().parents[1] / "configs" / "norm_stats.json"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Medie RGB su ImageNet (scala 0-1).
_IMAGENET_STD  = (0.229, 0.224, 0.225)  # Deviazioni standard RGB su ImageNet.


def _load_norm_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Carica mean/std dal file JSON, con fallback su statistiche ImageNet.

    Returns:
        Tupla `(mean, std)` dove `mean` e `std` sono a loro volta tuple di
        float (un valore per canale). I valori sono nella scala [0, 1] gia'
        adatta alle pipeline di albumentations/torchvision.

    Side effects:
        Emette un warning sul logger se il file `configs/norm_stats.json` non
        viene trovato (utile per diagnosticare un setup incompleto).
    """
    if _NORM_STATS_PATH.exists():
        with open(_NORM_STATS_PATH) as f:
            data = json.load(f)
        # Convertiamo le liste JSON in tuple per immutabilita': queste
        # statistiche non devono essere modificate dopo il caricamento.
        return tuple(data["mean"]), tuple(data["std"])
    logger.warning(
        "configs/norm_stats.json not found - falling back to ImageNet stats. "
        "Run scripts/compute_norm_stats.py to use train-set statistics.",
    )
    return _IMAGENET_MEAN, _IMAGENET_STD


# Caricamento "una tantum" all'import del modulo: le statistiche sono usate
# come default da `get_train_transform` e `get_eval_transform`.
_MEAN, _STD = _load_norm_stats()

# -----------------------------------------------------------------------------
# Parametri del rumore gaussiano per l'augmentation
# -----------------------------------------------------------------------------
# albumentations 2.x esprime il rumore gaussiano tramite `std_range` su scala
# [0, 1] (internamente moltiplica per 255). Vogliamo simulare un rumore con
# varianza in pixel-units fra 5 e 20:
#   std_pixel = sqrt(varianza)  => std_pixel in [sqrt(5), sqrt(20)]
#   std_normalizzato = std_pixel / 255
# Risultato: deviazioni standard normalizzate ~0.0088 e ~0.0175, ovvero un
# rumore "leggero" che simula la grana tipica delle CT senza distruggere le
# strutture anatomiche.
_GAUSS_STD_LO = math.sqrt(5) / 255   # ~ 0.0088
_GAUSS_STD_HI = math.sqrt(20) / 255  # ~ 0.0175

# -----------------------------------------------------------------------------
# Disponibilita' della libreria lungmask
# -----------------------------------------------------------------------------
# `lungmask` (pacchetto pip) fornisce una U-Net pre-addestrata per segmentare
# i polmoni. Se non e' installata (oppure se l'inferenza fallisce a runtime)
# usiamo come fallback un algoritmo classico basato su Otsu + morfologia.
# Questo import e' "soft" perche' la dipendenza e' opzionale: il progetto
# deve funzionare anche senza GPU/internet.
try:
    from lungmask import LMInferer as _LMInferer
    _LUNGMASK_AVAILABLE = True
except ImportError:
    _LUNGMASK_AVAILABLE = False
    logger.debug("lungmask not importable - Otsu fallback active for lung_mask()")


# =============================================================================
# Funzioni principali
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """Carica un'immagine PNG dal disco e la restituisce come array NumPy uint8.

    Gestisce automaticamente:
    - Immagini grayscale (forma HxW).
    - Immagini RGB (forma HxWx3) - convertite da BGR (default OpenCV) a RGB.
    - Immagini RGBA (4 canali) - canale alfa scartato.
    - Immagini a 16 bit (uint16) - degradate a 8 bit prendendo i byte alti
      (shift di 8 bit a destra), tecnica equivalente a un downsampling lineare
      del range dinamico.

    Args:
        path: Percorso del file immagine sul disco.

    Returns:
        Array NumPy di forma `(H, W)` per grayscale oppure `(H, W, 3)` per
        immagini a colori, con dtype `uint8` e canali in ordine RGB (non BGR).

    Raises:
        FileNotFoundError: Se OpenCV non riesce a leggere il file (path
            inesistente o formato non supportato).
    """
    # `IMREAD_UNCHANGED` preserva il numero originale di canali e la profondita'
    # in bit (necessario per gestire correttamente uint16/RGBA).
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # uint16 -> uint8: shift a destra di 8 bit. Equivalente a prendere il byte
    # piu' significativo. Le CT salvate a 16 bit sono ricondotte alla scala
    # 0-255 senza interpolazione (perdita di precisione accettabile per il
    # nostro task di classificazione).
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)

    # OpenCV restituisce le immagini a colori in BGR; convertiamo in RGB per
    # coerenza con il resto dell'ecosistema Python (PIL, matplotlib, torch).
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        # Immagine con canale alfa: scartato (per il nostro dominio la
        # trasparenza non e' informativa).
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img.astype(np.uint8)


def resize_with_padding(img: np.ndarray, target: int = 224) -> np.ndarray:
    """Ridimensiona `img` in un quadrato `target x target` preservando le proporzioni.

    Strategia: si calcola il fattore di scala per portare il lato piu' lungo a
    `target`, poi si centra l'immagine ridimensionata in un canvas nero
    `target x target`. Questo evita la distorsione che si avrebbe con un
    semplice `cv2.resize(img, (target, target))` su immagini non quadrate.

    Args:
        img: Array di input di forma `(H, W)` (grayscale) o `(H, W, C)`
            (multicanale), dtype `uint8`.
        target: Lato del quadrato di output in pixel. Default 224 (formato
            standard per i backbone tipo EfficientNet-B0).

    Returns:
        Array di forma `(target, target)` o `(target, target, C)`, dtype
        `uint8`, con l'immagine originale centrata e padding nero ai lati.
    """
    h, w = img.shape[:2]
    # Fattore di scala = target / lato_piu_lungo. Cosi' il lato piu' lungo
    # diventa esattamente `target` e l'altro lato resta proporzionalmente
    # piu' corto (verra' "paddato" con nero).
    scale = target / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    # INTER_LINEAR e' un buon compromesso fra qualita' e velocita' per immagini
    # mediche (INTER_CUBIC sarebbe leggermente migliore ma molto piu' lento).
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Creiamo il canvas nero della dimensione finale, rispettando la forma
    # (con o senza canali colore) dell'immagine originale.
    if img.ndim == 2:
        canvas = np.zeros((target, target), dtype=np.uint8)
    else:
        canvas = np.zeros((target, target, img.shape[2]), dtype=np.uint8)

    # Calcoliamo i pad top/left per centrare l'immagine. Il pad inferiore/destro
    # viene calcolato implicitamente dalla slice (potrebbe essere 1 pixel
    # maggiore in caso di parita' dispari, e questo va bene).
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applica CLAHE (equalizzazione adattiva con clipping) a un'immagine grayscale.

    CLAHE divide l'immagine in tile (8x8 per default), equalizza
    l'istogramma di ciascun tile e poi interpola bilinearmente i risultati ai
    confini per evitare artefatti a "blocchi". Il `clip_limit` limita la
    pendenza della CDF in ogni tile per non amplificare eccessivamente il
    rumore (cosa che farebbe l'equalizzazione globale classica). E' la tecnica
    standard per migliorare il contrasto in immagini mediche.

    Args:
        img: Immagine grayscale `(H, W)`, dtype `uint8`.
        clip_limit: Soglia di clipping del contrasto. Valori tipici 2.0-4.0:
            piu' alto = piu' contrasto ma anche piu' rumore.
        tile_grid: Numero di tile in cui suddividere l'immagine `(righe, colonne)`.
            (8, 8) e' un default robusto per immagini 224x224 o 512x512.

    Returns:
        Immagine grayscale `(H, W)`, dtype `uint8`, con contrasto migliorato.

    Raises:
        ValueError: Se l'immagine non e' 2-D (CLAHE non si applica direttamente
            a immagini a colori - bisognerebbe convertirle in LAB e applicarlo
            solo al canale L).
    """
    if img.ndim != 2:
        raise ValueError(
            f"apply_clahe expects a 2-D grayscale image, got shape {img.shape}"
        )
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def _otsu_fallback(gray: np.ndarray) -> np.ndarray:
    """Fallback "classico" per la segmentazione polmonare (senza deep learning).

    Pipeline:
        1. **Otsu thresholding**: trova automaticamente la soglia che separa
           sfondo (chiaro) da polmoni (scuri, perche' pieni d'aria nella CT).
        2. **Chiusura morfologica 15x15** (dilatazione + erosione): chiude i
           "buchi" interni ai polmoni causati da bronchi/vasi visibili.
        3. **Apertura morfologica 5x5** (erosione + dilatazione): rimuove
           piccoli artefatti isolati nello sfondo.
        4. **Connected components**: tiene solo le due componenti piu' grandi
           (sinistra + destra polmone) scartando rumore residuo.

    E' meno preciso di una U-Net dedicata ma e' deterministico, rapido e non
    richiede dipendenze esterne - utile come safety net.

    Args:
        gray: Immagine grayscale `(H, W)`, dtype `uint8`.

    Returns:
        Maschera binaria `(H, W)`, dtype `uint8`, con valori `{0, 255}`.
    """
    # Otsu: trova la soglia ottimale che massimizza la varianza inter-classe.
    # Soglia=0 e' un placeholder ignorato per via del flag THRESH_OTSU.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Chiusura: elimina i buchi neri dentro i polmoni (es. bronchi).
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Apertura: rimuove macchie bianche sparse fuori dai polmoni.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    # Etichetta tutte le componenti connesse. `stats` contiene metriche per
    # ogni componente (area, bbox, ecc.); l'etichetta 0 e' lo sfondo.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels < 2:
        # Caso degenere: nessuna componente trovata. Ritorniamo l'immagine
        # morfologicamente pulita senza filtri ulteriori.
        return opened
    areas = stats[1:, cv2.CC_STAT_AREA]          # Saltiamo l'etichetta 0 (sfondo).
    top_n = min(2, len(areas))
    # `argsort` ordina in modo crescente; prendiamo gli ultimi `top_n` (i piu'
    # grandi). Il `+1` rimette gli indici sullo spazio originale (rispetto al
    # quale avevamo escluso lo sfondo con `stats[1:]`).
    top_idx = np.argsort(areas)[-top_n:] + 1
    mask = np.zeros_like(opened)
    for idx in top_idx:
        mask[labels == idx] = 255
    return mask


def lung_mask(img: np.ndarray) -> np.ndarray:
    """Genera una maschera binaria dei polmoni usando lungmask (U-Net) o Otsu come fallback.

    Strategia "best effort": prova prima il modello U-Net pre-addestrato di
    `lungmask` (qualita' di segmentazione molto piu' alta) e, in caso di
    fallimento (dipendenza mancante o errore di inferenza), ricade
    sull'algoritmo classico basato su Otsu + morfologia.

    Args:
        img: Immagine grayscale `(H, W)` oppure RGB `(H, W, 3)`, dtype `uint8`.

    Returns:
        Maschera binaria `(H, W)`, dtype `uint8`, con valori `{0, 255}`
        (255 = pixel appartenente ai polmoni).

    Side effects:
        Logga un warning se lungmask fallisce (per facilitare il debug).
    """
    # Lungmask lavora su immagini grayscale. Se ricevo un'immagine RGB la
    # converto in scala di grigi.
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if _LUNGMASK_AVAILABLE:
        try:
            inferer = _LMInferer()
            # Lungmask si aspetta un "volume" (D, H, W). Per una singola slice
            # 2D aggiungiamo una dimensione "fittizia" di profondita' D=1.
            vol = gray[np.newaxis, ...]   # (1, H, W) - pseudo-volume monoslice
            # Il modello restituisce etichette {0=background, 1=polmone sx,
            # 2=polmone dx}. Le binarizziamo a {0, 255} con `> 0`.
            seg = inferer.apply(vol)
            return (seg[0] > 0).astype(np.uint8) * 255
        except Exception as exc:
            # Catch generico: vogliamo che il fallback funzioni *sempre*,
            # qualunque sia la causa dell'errore (modello mancante, CUDA OOM,
            # input malformato, ecc.).
            logger.warning("lungmask inference failed (%s); using Otsu fallback", exc)

    return _otsu_fallback(gray)


def apply_lung_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Azzera i pixel esterni alla maschera polmonare.

    Operazione: `output = img * (mask > 0)`. Mantiene invariati i pixel "dentro"
    i polmoni e mette a zero quelli "fuori". Utile per rimuovere artefatti di
    sfondo (scritte, barre laterali) che potrebbero indurre il modello a
    imparare scorciatoie spurie.

    Args:
        img: Immagine `(H, W)` o `(H, W, C)`, dtype `uint8`.
        mask: Maschera binaria `(H, W)`, dtype `uint8`. Qualunque valore > 0
            e' considerato "dentro" il polmone.

    Returns:
        Immagine mascherata della stessa forma di `img`, dtype `uint8`.
    """
    # Binarizziamo la maschera: il valore esatto (es. 255 vs 1) non ci interessa,
    # ci basta sapere "dentro/fuori".
    binary = (mask > 0).astype(np.uint8)
    if img.ndim == 3:
        # Per immagini multicanale aggiungiamo l'asse "canali" alla maschera
        # cosi' che il broadcasting NumPy moltiplichi tutti i canali allo stesso
        # modo (mask (H,W,1) * img (H,W,C) -> (H,W,C)).
        binary = binary[:, :, np.newaxis]
    return (img * binary).astype(np.uint8)


def build_three_channel(img_gray: np.ndarray) -> np.ndarray:
    """Costruisce un'immagine a 3 canali a partire da una slice CT grayscale.

    Idea: i backbone pre-addestrati (EfficientNet, DenseNet, ...) si aspettano
    in input 3 canali RGB. Avendo una sola slice grayscale possiamo:

    - **Opzione semplice**: replicare lo stesso canale 3 volte (informazione
      ridondante: la rete vede solo una "vista" dell'immagine).
    - **Opzione adottata qui**: comporre un "RGB sintetico" con tre viste
      complementari della stessa slice:

      * canale 0: grayscale originale,
      * canale 1: CLAHE con `clip_limit=2.0` (contrasto moderato),
      * canale 2: CLAHE con `clip_limit=4.0` (contrasto aggressivo).

    Cosi' la rete riceve in input piu' informazione "preprocessing-aware"
    senza dover modificare l'architettura del backbone.

    Args:
        img_gray: Immagine grayscale `(H, W)`, dtype `uint8`.

    Returns:
        Array `(H, W, 3)`, dtype `uint8`, con i tre canali descritti sopra.

    Raises:
        ValueError: Se l'immagine non e' 2-D.
    """
    if img_gray.ndim != 2:
        raise ValueError(
            f"build_three_channel expects a 2-D array, got shape {img_gray.shape}"
        )
    ch1 = apply_clahe(img_gray, clip_limit=2.0)
    ch2 = apply_clahe(img_gray, clip_limit=4.0)
    # `np.stack` con `axis=-1` impila lungo l'ultimo asse (= asse canali).
    return np.stack([img_gray, ch1, ch2], axis=-1)


# =============================================================================
# Pipeline di augmentation (albumentations)
# =============================================================================

def get_train_transform(
    mean: tuple[float, ...] = _MEAN,
    std: tuple[float, ...] = _STD,
) -> A.Compose:
    """Restituisce la pipeline augmentation + normalizzazione per il training.

    Trasformazioni applicate (ognuna con probabilita' indicata):

    - `HorizontalFlip(p=0.5)`: flip orizzontale - lecito per CT del torace
      (un polmone destro "diventa" un polmone sinistro, ma il task non
      richiede di distinguerli).
    - `Rotate(limit=10, p=0.5)`: rotazione casuale fra -10 e +10 gradi.
      Limite stretto perche' rotazioni grandi sarebbero anatomicamente
      irrealistiche.
    - `RandomBrightnessContrast(p=0.5)`: piccola variazione di
      luminosita'/contrasto, simula differenze fra acquisizioni di scanner
      diversi.
    - `Affine(scale=(0.9, 1.1), p=0.5)`: zoom casuale +/- 10%, simula
      distanze paziente-scanner leggermente diverse.
    - `GaussNoise(...)`: rumore gaussiano leggero, simula la grana tipica
      delle ricostruzioni CT.
    - `Normalize(mean, std)`: sottrae la media e divide per la std (per canale).
    - `ToTensorV2()`: converte l'array NumPy `(H, W, C)` uint8 in tensore
      PyTorch `(C, H, W)` float32.

    Vincoli di dominio che NON appaiono di proposito:
    - **No VerticalFlip**: la gravita' "sa" dove sono cuore e diaframma; un
      flip verticale produrrebbe immagini impossibili.
    - **No HueSaturationValue**: le CT sono in scala di grigi, non ha senso
      perturbare la saturazione.
    - **No rotazioni > 15 gradi**.

    Args:
        mean: Tupla con la media per canale (default: statistiche del
            training set).
        std: Tupla con la deviazione standard per canale.

    Returns:
        Oggetto `albumentations.Compose` richiamabile come `transform(image=arr)`.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.Affine(scale=(0.9, 1.1), p=0.5),
            A.GaussNoise(
                std_range=(_GAUSS_STD_LO, _GAUSS_STD_HI), p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_eval_transform(
    mean: tuple[float, ...] = _MEAN,
    std: tuple[float, ...] = _STD,
) -> A.Compose:
    """Restituisce la pipeline (solo normalizzazione) per validation e test.

    In fase di valutazione NON applichiamo augmentation: vogliamo che il
    risultato sia deterministico e che le metriche riflettano il
    comportamento del modello sull'immagine "vera", senza varianza random.

    Args:
        mean: Tupla con la media per canale (default: statistiche del
            training set).
        std: Tupla con la deviazione standard per canale.

    Returns:
        Oggetto `albumentations.Compose` che applica solo `Normalize` +
        `ToTensorV2`.
    """
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
