"""Cattura uno screenshot della demo Streamlit in esecuzione, con un sample COVID caricato.

Scopo dello script
------------------
Helper interno per rigenerare `docs/figures/streamlit_demo.png` (lo
screenshot mostrato nel README e nel report). Usa Playwright per:
1. Aprire un browser Chromium headless.
2. Navigare a `http://localhost:8501` (la demo Streamlit deve essere gia'
   in esecuzione separatamente).
3. Caricare una slice COVID di esempio tramite il file input nascosto.
4. Attendere il rendering della predizione.
5. Salvare uno screenshot full-page.

Importante
----------
Lo script ASSUME che Streamlit sia gia' in ascolto su porta 8501. Avvia
prima la demo con:

    streamlit run app/demo.py

Poi lancia questo script in un secondo terminale.

Uso
---
    PYTHONPATH=. python scripts/screenshot_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Path "ancorati" al filesystem: calcolati una volta sola all'import.
# `__file__` -> path dello script; `parents[1]` -> root del progetto.
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sarscov2-ctscan-dataset" / "COVID" / "Covid (1).png"
OUT = ROOT / "docs" / "figures" / "streamlit_demo.png"
URL = "http://localhost:8501"


def main() -> int:
    """Entry point CLI.

    Returns:
        Exit code: 0 se tutto OK, 1 se il sample image manca.
    """
    if not SAMPLE.exists():
        # Stampa diretta su stderr: questo script e' un helper, non usa il
        # logger del progetto (semplicita').
        print(f"Missing sample image: {SAMPLE}", file=sys.stderr)
        return 1

    # `sync_playwright()` come context manager: garantisce la chiusura
    # del browser anche in caso di eccezione.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Viewport ampio e device_scale_factor=2 per screenshot ad alta DPI
        # (retina-friendly), adatti a un README.
        ctx = browser.new_context(viewport={"width": 1280, "height": 1500}, device_scale_factor=2)
        page = ctx.new_page()

        # Carica la pagina e aspetta che la rete sia "idle" (= Streamlit
        # ha finito di hydrare il DOM).
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        # Aspetto che il titolo della demo sia visibile - significa che
        # Streamlit ha completato il primo render.
        page.wait_for_selector("text=COVID-19 CT-Scan classification", timeout=30_000)

        # Upload del sample image: Streamlit nasconde il <input type="file">
        # ma e' comunque interattivo dal DOM.
        page.set_input_files("input[type='file']", str(SAMPLE))

        # Attendi che la sezione di output sia comparsa (la predizione
        # appare DOPO l'upload, perche' Streamlit fa rerun del flow).
        page.wait_for_selector("text=Predicted class", timeout=30_000)
        page.wait_for_selector("text=P(COVID)",        timeout=30_000)
        # Sleep esplicito di 1.5s: dopo che gli elementi sono presenti, le
        # immagini Grad-CAM possono richiedere ancora qualche frame per il
        # rendering completo. Workaround pragmatico.
        time.sleep(1.5)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        # `full_page=True` cattura anche la parte sotto il viewport
        # (scrolling) - utile perche' la demo Streamlit e' alta.
        page.screenshot(path=str(OUT), full_page=True)
        print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")

        browser.close()
    return 0


# `raise SystemExit(...)` invece di `sys.exit(...)`: idiomatico e piu'
# pulito quando lo script puo' uscire con codici diversi.
if __name__ == "__main__":
    raise SystemExit(main())
