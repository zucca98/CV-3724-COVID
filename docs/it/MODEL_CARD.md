# Model Card - Classificazione COVID-CT (SVM + EfficientNet-B0)

## Dettagli del modello

- **Sviluppato da:** Alex Zuccolin (EPICODE - corso di Computer Vision, 2026).
- **Tipo di modello:** due classificatori binari paralleli per slice di TAC
  del torace.
  - **Baseline classica:** SVM con kernel RBF su PCA-200 di feature
    HOG + LBP + GLCM.
  - **Modello deep:** EfficientNet-B0 (timm, pretrained su ImageNet) con head
    personalizzata (Dropout 0.5 -> Linear 1280->2), fine-tuning a due fasi.
- **Framework:** scikit-learn 1.8, PyTorch 2.x, timm 1.0, albumentations 2.x.
- **Input:** singola slice assiale di TAC del torace in formato PNG (qualsiasi
  dimensione e aspect ratio).
- **Output:** probabilita' della classe *COVID* in [0, 1] e label binaria.
- **Licenza:** MIT (codice); dati di training CC BY 4.0 (Soares et al., 2020).
- **Versione:** 0.1.0.

## Uso previsto

- **Uso primario previsto:** didattica accademica e benchmarking sul dataset
  SARS-CoV-2 CT-Scan.
- **Fuori scope:** qualsiasi applicazione clinica, di triage o diagnostica.
  Il sistema non e' stato valutato secondo i quadri regolatori richiesti per
  il software medicale (marcatura CE, FDA 510(k)).

## Fattori

- **Fattori rilevanti:** produttore dello scanner, spessore della slice, eta'
  / sesso / comorbidita' del paziente, paese di acquisizione, finestra di
  acquisizione dell'immagine, presenza di artefatti di imaging.
- **Fattori di valutazione:** nessuno dei suddetti e' annotato nel dataset,
  quindi non e' stato possibile eseguire valutazioni per sottogruppi.

## Metriche

Test set (n = 373, bilanciato, stratificato, seed 42, nessun raggruppamento
per paziente).

| Metrica | SVM | Random Forest | EfficientNet-B0 |
|---------|----:|--------------:|----------------:|
| Accuracy | 0.962 | 0.887 | 0.933 |
| Recall (COVID) | 0.968 | 0.878 | 0.920 |
| Specificita' | 0.957 | 0.897 | 0.946 |
| F1 | 0.963 | 0.887 | 0.933 |
| AUC-ROC | 0.996 | 0.952 | 0.981 |
| Brier | 0.024 | 0.103 | n/d |
| ECE (10 bin) | 0.027 | 0.128 | n/d |

I valori di EfficientNet-B0 derivano dal run completo di fine-tuning a 40
epoche (10 solo testa + 30 complete) su una GPU RTX 3080 locale, valutato alla
soglia di default 0.5. Brier/ECE non sono stati estratti per il modello deep;
il temperature scaling ha trovato T = 1.002, indice di un modello gia' ben
calibrato. La baseline SVM resta il modello piu' forte su questo dataset
(AUC 0.996 vs 0.981).

La soglia di decisione per l'operativita' di tipo deployment e' tarata per
garantire recall COVID >= 0.95 sul validation set calibrato (soglia scelta
0.358). Sul test set held-out quella soglia produce recall = 0.931 - il target
>= 0.95 e' raggiunto in validation ma non pienamente in test, segno che la
soglia non generalizza perfettamente. Vedere
`docs/technical_analysis.pdf` § Metodologia.

## Dati di training

- **Dataset:** SARS-CoV-2 CT-Scan (Soares et al., 2020), 2 481 immagini PNG,
  ospedali brasiliani, due classi (COVID / non-COVID).
- **Split:** 70 / 15 / 15 stratificati per etichetta, `random_state=42`.
- **Nessun identificatore paziente** disponibile - vedere *Avvertenze* sotto.

## Dati di valutazione

Test set held-out (15%) dello stesso dataset.

## Considerazioni etiche

- Le predizioni non devono influenzare decisioni cliniche.
- Le mappe di salienza (Grad-CAM) sono fornite per l'interpretabilita' ma non
  sostituiscono una spiegabilita' di livello clinico.
- Il modello non e' stato sottoposto ad audit di fairness sui sottogruppi
  demografici perche' i metadati rilevanti sono assenti dal dataset.

## Avvertenze e raccomandazioni

- **Leakage paziente:** l'assenza di ID paziente significa che slice dello
  stesso paziente possono attraversare diversi split. Le metriche riportate
  potrebbero sovrastimare la generalizzazione.
- **Distribution shift:** valutazione eseguita solo su dati di ospedali
  brasiliani. Le prestazioni su scanner/popolazioni differenti sono ignote.
- **Raccomandazione:** ogni scenario di deployment dovrebbe ri-validare con
  uno split stratificato per paziente e una valutazione prospettica su una
  coorte rappresentativa.
