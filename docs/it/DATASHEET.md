# Datasheet - Dataset SARS-CoV-2 CT-Scan (uso in questo progetto)

Questo datasheet documenta in che modo il dataset **SARS-CoV-2 CT-Scan**
viene utilizzato in questo progetto; non sostituisce la documentazione
originale di Soares et al.

## Motivazione

- **Perche' e' stato creato il dataset?** Per consentire ricerca rapida sul
  rilevamento da TAC della polmonite da SARS-CoV-2 nel pieno della pandemia.
- **Chi lo ha creato?** Eduardo Soares, Plamen Angelov e colleghi
  (Lancaster University e ospedali brasiliani); distribuito tramite Kaggle.
- **Chi lo ha finanziato?** Come riportato in Soares et al. (2020) - vedere
  riferimenti.
- **Uso in questo progetto:** benchmark di classificazione binaria per un
  progetto accademico del corso di Computer Vision.

## Composizione

- **Cosa rappresenta ogni istanza?** Una singola slice assiale di TAC in
  formato PNG, scala di grigi o RGB, a risoluzioni variabili (tipicamente
  200-500 px).
- **Quante istanze?** 2 481 PNG, due cartelle: `COVID/` (1 252) e
  `non-COVID/` (1 229).
- **Bilanciamento delle classi:** approssimativamente bilanciato
  (50.5% / 49.5%).
- **Metadati paziente:** **assenti.** Non sono inclusi ID paziente, modello
  di scanner, spessore della slice, eta', sesso o comorbidita'.
- **Split:** la release originale e' non suddivisa; questo progetto applica
  internamente uno split stratificato 70 / 15 / 15 con `random_state=42`.

## Processo di raccolta

- **Origine:** due ospedali di San Paolo, Brasile (secondo il paper
  originale).
- **Acquisizione:** raccolta retrospettiva di TAC cliniche.
- **Campionamento:** convenience sampling - non necessariamente
  rappresentativo della popolazione generale.
- **Periodo:** 2020 (fase iniziale della pandemia).

## Preprocessing / pulizia / etichettatura

- I curatori originali hanno etichettato le immagini a livello di slice
  tramite esame clinico.
- Questo progetto applica una propria pipeline di preprocessing (vedere
  `src/preprocessing.py`): resize con padding a 224 x 224, lung masking
  opzionale (U-Net o fallback Otsu), composito CLAHE a 3 canali,
  augmentation.
- Non e' stata effettuata alcuna ri-etichettatura.

## Usi

- **Usi appropriati:** didattica, benchmarking, esperimenti metodologici.
- **Usi inappropriati:** supporto a decisioni cliniche, triage, dossier
  regolatori. Il dataset ha limiti noti (assenza di ID paziente, singola
  geografia) che invaliderebbero qualsiasi claim clinico.

## Distribuzione

- Disponibile su Kaggle:
  <https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>
- Licenza: **CC BY 4.0**, con attribuzione obbligatoria a Soares et al.
  (2020).

## Manutenzione

- Questo progetto blocca il dataset replicando l'hash intero del nome di
  ciascun file nei CSV di split (`data/processed/{train,val,test}.csv`).
- Non viene monitorata alcuna ri-etichettatura upstream. Se il dataset viene
  aggiornato, gli split devono essere rigenerati con
  `scripts/prepare_data.py`.

## Limiti e rischi noti

1. **Nessun identificatore paziente** => slice dello stesso paziente possono
   finire in split diversi, gonfiando le metriche.
2. **Omogeneita' geografica** (Brasile) => scarse garanzie di
   generalizzazione.
3. **Metadati demografici assenti** => impossibile fare audit di fairness.
4. **Eterogeneita' di acquisizione ignota** => non si possono escludere
   shortcut basati su modello di scanner / spessore della slice.
5. **Qualita' delle annotazioni:** le etichette upstream sono binarie a
   livello di slice; non sono fornite maschere a livello di regione della
   patologia.

## Riferimento

> Soares, E., Angelov, P., Biaso, S., Froes, M. H., & Abe, D. K. (2020).
> *SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for
> SARS-CoV-2 identification.* medRxiv.
> <https://doi.org/10.1101/2020.04.24.20078584>
