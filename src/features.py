"""Estrazione di feature handcrafted (LBP + GLCM + HOG) per la pipeline classica.

Scopo del file
--------------
Questo modulo implementa il braccio "classico" della pipeline: a partire da
una slice CT in scala di grigi calcola un vettore numerico di feature
*handcrafted* (cioe' progettate a mano dal ricercatore, non apprese da una
rete neurale). Questo vettore viene poi dato in pasto a un classificatore
SVM/RandomForest (vedi `src/models/classical.py`).

Le tre famiglie di feature usate sono:

1. **LBP** (Local Binary Pattern) - cattura la *texture* locale: per ogni
   pixel confronta i vicini in un cerchio e codifica il risultato come
   numero binario; l'istogramma dei codici riassume la texture dell'intera
   immagine. Ottimo per discriminare "ground-glass opacity" tipico del COVID.

2. **GLCM** (Gray-Level Co-occurrence Matrix) - cattura la *texture* a piu'
   alto livello: misura quanto spesso coppie di pixel con certi livelli di
   grigio appaiono a distanza e orientamento fissati. Da questa matrice si
   calcolano metriche statistiche (contrasto, omogeneita', ecc.).

3. **HOG** (Histogram of Oriented Gradients) - cattura la *forma*: per
   piccole celle calcola l'istogramma delle orientazioni del gradiente.
   Storica feature di Dalal & Triggs per pedestrian detection, generalizza
   bene a forme anatomiche.

Concatenando i tre vettori si ottiene un descrittore robusto e
interpretabile (~26 mila feature per immagine 224x224).

Integrazione
------------
- Importato da `scripts/extract_features.py` (batch su tutto il dataset).
- Usato indirettamente da `src/models/classical.py` (allenamento SVM/RF).
- Dipende da `src/preprocessing.apply_lung_mask` per filtrare la regione
  polmonare prima dell'estrazione.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from src.preprocessing import apply_lung_mask
from src.utils import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Parametri GLCM
# -----------------------------------------------------------------------------
# Per ogni cella (distanza, angolo) della GLCM calcoliamo queste 6 statistiche
# canoniche. Si veda Haralick et al. (1973) per le definizioni formali.
_GLCM_PROPS: list[str] = [
    "contrast",      # somma pesata di (i-j)^2 - quanto i pixel vicini differiscono
    "dissimilarity", # somma pesata di |i-j| - variante L1 del contrasto
    "homogeneity",   # somma di 1/(1+(i-j)^2) - alta su texture uniformi
    "ASM",           # Angular Second Moment - somma dei quadrati = "uniformita'"
    "energy",        # sqrt(ASM)
    "correlation",   # correlazione lineare fra livelli grigio adiacenti
]

# Quantizzazione dei livelli di grigio per la GLCM:
# - immagine originale 256 livelli (uint8)
# - shift a destra di 2 bit  =>  diviso per 4  =>  64 livelli
# Vantaggio: la matrice GLCM scende da 256x256 (=65 536 celle) a 64x64
# (=4 096 celle), 16 volte piu' piccola e piu' veloce da calcolare,
# mantenendo abbastanza risoluzione per le metriche di texture.
_GLCM_LEVELS = 64
_GLCM_SHIFT = 2


def extract_lbp(
    img_gray: np.ndarray,
    P: int = 24,
    R: int = 3,
) -> np.ndarray:
    """Calcola l'istogramma LBP "uniforme" normalizzato.

    "Uniforme" significa che si considerano solo i pattern LBP con al massimo
    2 transizioni 0-1 (es. 00111100 e' uniforme, 01010101 no). Tutti i
    pattern non uniformi finiscono in un unico bin "spazzatura", riducendo
    cosi' la dimensionalita' dell'istogramma a `P + 2` bin (P pattern
    uniformi + 1 bin per il pattern "tutto zero" + 1 bin per quelli non
    uniformi). Questa variante e' robusta al rumore e quasi invariante per
    rotazione.

    Args:
        img_gray: Immagine grayscale `(H, W)`, dtype `uint8`.
        P: Numero di punti vicini campionati sul cerchio. Default 24 = 24
            campioni equidistanti.
        R: Raggio del cerchio di campionamento, in pixel. Default 3.

    Returns:
        Istogramma normalizzato di forma `(P + 2,)` (= 26 bin di default).
        I valori sommano a 1, eccetto per un epsilon di stabilita' numerica.
    """
    lbp = local_binary_pattern(img_gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    # Aggiungiamo 1e-9 al denominatore per evitare divisione per zero nel caso
    # patologico di un'immagine totalmente nera (mask piena di zeri).
    return hist.astype(np.float64) / (hist.sum() + 1e-9)


def extract_glcm(
    img_gray: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> np.ndarray:
    """Calcola le proprieta' GLCM e restituisce un vettore feature appiattito.

    Per ogni coppia (distanza, angolo) costruisce la GLCM e ne estrae le 6
    proprieta' definite in `_GLCM_PROPS`. Il risultato viene poi appiattito
    in un vettore 1-D.

    Args:
        img_gray: Immagine grayscale `(H, W)`, dtype `uint8`.
        distances: Distanze (in pixel) per il calcolo della co-occorrenza.
            Default `[1, 2, 3, 4]` per catturare texture a piu' scale.
        angles: Direzioni (in radianti) della co-occorrenza.
            Default `[0, pi/4, pi/2, 3*pi/4]` ovvero 0deg, 45deg, 90deg, 135deg
            (le 8 direzioni "specchiate" coincidono perche' usiamo
            `symmetric=True`).

    Returns:
        Array 1-D `float64` di lunghezza
        `len(_GLCM_PROPS) * len(distances) * len(angles)`.
        Con i default: 6 * 4 * 4 = 96 feature.
    """
    if distances is None:
        distances = [1, 2, 3, 4]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Quantizzazione: da 256 livelli a 64. `>> 2` equivale a `// 4`.
    # Importante mantenere dtype uint8 perche' `graycomatrix` lo richiede.
    img_q = (img_gray >> _GLCM_SHIFT).astype(np.uint8)
    glcm = graycomatrix(
        img_q,
        distances=distances,
        angles=angles,
        levels=_GLCM_LEVELS,
        symmetric=True,  # GLCM(i,j) += GLCM(j,i) -> invariante al verso
        normed=True,     # divide ogni cella per il totale -> probabilita'
    )
    # `graycoprops(glcm, prop)` restituisce un array (n_distances, n_angles);
    # `.ravel()` lo appiattisce. Concateniamo tutte le proprieta' in un
    # singolo vettore.
    return np.concatenate(
        [graycoprops(glcm, prop).ravel() for prop in _GLCM_PROPS]
    ).astype(np.float64)


def extract_hog(img_gray: np.ndarray) -> np.ndarray:
    """Calcola il descrittore HOG (Histogram of Oriented Gradients).

    Parametri scelti (i default di skimage):
    - `orientations=9`: 9 bin di orientazione (~ 20deg ciascuno) -> 0deg..180deg
    - `pixels_per_cell=(8, 8)`: ogni cella e' 8x8 pixel
    - `cells_per_block=(2, 2)`: ogni blocco e' 2x2 celle
    - `block_norm="L2-Hys"`: normalizzazione L2 seguita da clipping e
      ri-normalizzazione (Dalal & Triggs).

    Calcolo della dimensione output per un'immagine 224x224:
        celle per lato = 224 / 8 = 28
        blocchi per lato = 28 - 2 + 1 = 27
        feature per blocco = 2 * 2 * 9 = 36
        totale = 27 * 27 * 36 = 26 244

    Args:
        img_gray: Immagine grayscale `(H, W)`, dtype `uint8`.

    Returns:
        Vettore HOG 1-D `float64`, lunghezza dipendente da H, W (vedi sopra).
    """
    fd = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,  # restituisce array 1-D anziche' tensore multi-dim
    )
    return fd.astype(np.float64)


def extract_handcrafted_features(
    img_gray: np.ndarray,
    lung_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Concatena LBP + GLCM + HOG in un singolo vettore feature.

    Layout del vettore finale (per immagine 224x224 con default):

        [ LBP (26) | GLCM (96) | HOG (26 244) ]  ->  26 366 feature

    L'HOG domina dimensionalmente, motivo per cui in `classical.py` applichiamo
    PCA a 200 componenti prima del classificatore: senza PCA il kernel
    dell'SVM avrebbe un costo computazionale proibitivo.

    Args:
        img_gray: Immagine grayscale `(H, W)`, dtype `uint8`.
        lung_mask: Maschera binaria opzionale `(H, W)`. Se fornita, i pixel
            *fuori* dai polmoni vengono azzerati prima del calcolo delle
            feature, in modo che artefatti di sfondo non influenzino i
            descrittori.

    Returns:
        Vettore 1-D `float64` con la concatenazione delle tre famiglie di
        feature.
    """
    if lung_mask is not None:
        # Applichiamo la maschera *prima* di estrarre le feature: cosi'
        # l'istogramma LBP, la GLCM e l'HOG vedono solo la regione polmonare.
        img_gray = apply_lung_mask(img_gray, lung_mask)

    return np.concatenate(
        [
            extract_lbp(img_gray),
            extract_glcm(img_gray),
            extract_hog(img_gray),
        ]
    )
