"""Demo Streamlit: upload di una slice CT del torace -> predizione SVM + overlay.

Scopo dell'app
--------------
E' la **demo interattiva** del progetto: una mini web-app (Streamlit) che
permette a un utente di caricare un'immagine PNG/JPG di una slice CT del
torace e ottenere immediatamente:
- La classe predetta (COVID / non-COVID).
- La probabilita' di COVID `P(COVID)`.
- La soglia di decisione (regolabile via slider).
- L'immagine pre-processata (composite 3 canali CLAHE).
- La lung mask binaria (se abilitata).

Modello usato: l'**SVM su feature handcrafted** allenato da
`scripts/train_classical.py`. La scelta dell'SVM (e non del deep learning)
ha motivazioni pratiche:
- L'SVM gira al 100% su CPU -> demo deployabile ovunque.
- Modello completamente allenato e versionato nel repo (`models/classical/svm.pkl`).
- Il deep learning richiederebbe GPU e file `.pth` non committati nel repo
  (vedi `.gitignore`).

La lung mask viene calcolata al volo con l'algoritmo Otsu fallback (no
dipendenze GPU).

Avvio
-----
    streamlit run app/demo.py

Poi aprire `http://localhost:8501` nel browser.

ATTENZIONE
----------
Demo accademica - **NON E' UN DISPOSITIVO MEDICO**. Non usare per
diagnosi, triage o decisioni cliniche.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.features import extract_handcrafted_features
from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    load_image,
    lung_mask,
    resize_with_padding,
)

# Path "ancorati" alla struttura del progetto. `Path(__file__).resolve()`
# garantisce un path assoluto e canonico anche se Streamlit lancia lo
# script da una working directory diversa.
ROOT = Path(__file__).resolve().parents[1]
SVM_PATH = ROOT / "models" / "classical" / "svm.pkl"

# Disclaimer medico-legale: mostrato sempre in cima alla demo. Importante
# perche' progetti come questo possono indurre falso senso di affidabilita'.
MEDICAL_DISCLAIMER = """
> ! **NOT FOR CLINICAL USE.** This demo is an academic exercise. Do not use it
> for medical diagnosis, triage, or treatment decisions.
"""


@st.cache_resource
def _load_svm():
    """Carica il modello SVM dal disco, una sola volta per sessione Streamlit.

    `@st.cache_resource` e' il decoratore Streamlit per oggetti pesanti
    (modelli ML, connessioni DB, ...): evita di ricaricare il file .pkl
    ad ogni interazione dell'utente.

    Returns:
        Pipeline sklearn (scaler+pca+svm) caricata, oppure `None` se il
        file non esiste (l'utente non ha ancora addestrato il modello).
    """
    if not SVM_PATH.exists():
        return None
    return joblib.load(SVM_PATH)


def _preprocess(uploaded_bytes: bytes, target: int = 224, apply_mask: bool = True):
    """Esegue la stessa pipeline di preprocessing del training su un'immagine caricata.

    Workaround: `load_image` accetta solo Path (richiama OpenCV
    `cv2.imread`), quindi salviamo i bytes ricevuti da Streamlit in un file
    temporaneo, lo leggiamo, e poi lo eliminiamo.

    Args:
        uploaded_bytes: Contenuto raw del file caricato dall'utente.
        target: Dimensione del resize quadrato. Default 224.
        apply_mask: Se applicare la lung mask. Default True.

    Returns:
        Tupla `(img3, gray, mask)`:
        - `img3`: array `(target, target, 3)` uint8 (composite CLAHE).
        - `gray`: array `(target, target)` uint8 (grayscale post-mask).
        - `mask`: array `(target, target)` uint8 o `None` se mask disabilitata.
    """
    # File temporaneo in root: nome con prefisso `.` per nasconderlo a `ls`.
    # `unlink(missing_ok=True)` lo rimuove subito dopo la lettura per non
    # lasciare spazzatura.
    tmp = ROOT / ".streamlit_tmp.png"
    tmp.write_bytes(uploaded_bytes)
    img = load_image(tmp)
    tmp.unlink(missing_ok=True)
    gray = img if img.ndim == 2 else img.mean(axis=2).astype(np.uint8)
    gray = resize_with_padding(gray, target=target)
    mask = lung_mask(gray) if apply_mask else None
    if mask is not None:
        gray = apply_lung_mask(gray, mask)
    img3 = build_three_channel(gray)
    return img3, gray, mask


def main() -> None:
    """Entry point Streamlit: definisce la UI dell'app.

    Layout:
        - Titolo + disclaimer in alto.
        - Sidebar: toggle mask, slider soglia.
        - Body principale: file uploader.
        - Dopo upload: due colonne (immagine preprocessata | predizione)
          + opzionale lung mask sotto.
    """
    st.set_page_config(page_title="COVID-CT Demo", layout="centered")
    st.title("COVID-19 CT-Scan classification - academic demo")
    st.markdown(MEDICAL_DISCLAIMER)

    svm = _load_svm()
    if svm is None:
        # Errore esplicito + istruzioni invece di un crash silente:
        # l'utente capisce subito cosa deve fare.
        st.error(
            "SVM model not found at `models/classical/svm.pkl`.\n"
            "Run `scripts/train_classical.py` first."
        )
        return

    # Sidebar: i due controlli "esperti". Defaults sensati per uso casual.
    apply_mask = st.sidebar.checkbox("Apply lung mask", value=True)
    # Slider con step 0.01 per controllo fine sulla soglia (default 0.5).
    threshold  = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    # File uploader Streamlit: accetta solo le estensioni indicate.
    uploaded = st.file_uploader("Upload a chest-CT slice (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Awaiting an image upload ...")
        return

    raw_bytes = uploaded.read()
    img3, gray, mask = _preprocess(raw_bytes, apply_mask=apply_mask)

    # Estrazione feature + predizione SVM. `reshape(1, -1)` converte il
    # vettore 1D in matrice (1, n_features) come si aspetta sklearn.
    feats = extract_handcrafted_features(gray, lung_mask=mask if apply_mask else None)
    proba = svm.predict_proba(feats.reshape(1, -1))[0, 1]
    pred = "COVID" if proba >= threshold else "non-COVID"

    # Layout a due colonne. `st.columns(2)` divide lo spazio orizzontale.
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preprocessed input")
        # `use_container_width=True` ridimensiona l'immagine al contenitore.
        st.image(img3, caption="3-channel CLAHE composite", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        # `st.metric` rende un widget "big number" per dare risalto.
        st.metric(label="Predicted class", value=pred)
        st.metric(label="P(COVID)", value=f"{proba:.3f}")
        st.write(f"Decision threshold: **{threshold:.2f}**")

    # Mostriamo la maschera SOTTO le due colonne, solo se attiva (verbosita'
    # condizionale: se l'utente l'ha disabilitata via sidebar, evitiamo
    # output inutile).
    if mask is not None:
        st.subheader("Lung mask")
        # `clamp=True` riscala automaticamente i valori (0/255) in [0, 1]
        # per la visualizzazione (Streamlit a volte si confonde con uint8).
        st.image(mask, caption="Binary lung mask", clamp=True, use_container_width=True)


# Streamlit chiama questo script come modulo (`runpy`), quindi __name__ ==
# '__main__'. Manteniamo il guard per coerenza con il resto del progetto.
if __name__ == "__main__":
    main()
