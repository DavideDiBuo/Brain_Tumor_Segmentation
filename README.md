# Brain Tumor Segmentation con CNN (U-Net) su BraTS 2020 — Report (raw markdown)

## 1) Introduzione: reti neurali, CNN e segmentazione

### 1.1 Reti neurali (Neural Networks)
Una **rete neurale artificiale** è un modello composto da strati (layer) di unità computazionali che applicano trasformazioni non lineari ai dati. In forma semplificata:

- **Input**: vettore/tensore di caratteristiche (feature).
- **Layer**: trasformazioni lineari (pesi) + **funzioni di attivazione** (ReLU, sigmoid, ecc.) per introdurre non linearità.
- **Output**: predizione (classificazione, regressione, ecc.).
- **Addestramento**: si minimizza una **funzione di perdita (loss)** tramite ottimizzazione (es. **Adam**) usando **backpropagation**.

Nel deep learning, aumentando profondità e capacità del modello si riescono a catturare pattern complessi, a patto di avere dati e regolarizzazione adeguati.

### 1.2 Convolutional Neural Networks (CNN)
Le **CNN** sono reti specializzate per dati strutturati come immagini. Al posto di collegamenti fully-connected, usano **convoluzioni**:

- Un kernel (filtro) scorre sull’immagine e produce feature map.
- Le convoluzioni sfruttano **località** e **condivisione dei pesi**, riducendo parametri e migliorando generalizzazione.
- Tipicamente si alternano:
  - **Conv2D + ReLU**
  - **Pooling** (es. MaxPooling) per ridurre risoluzione e aumentare campo recettivo
  - (Opzionale) **Dropout / BatchNorm** per stabilità e regolarizzazione

### 1.3 Segmentazione (semantic segmentation)
La **segmentazione** consiste nel predire **una classe per ogni pixel (o voxel)**. In ambito medico (MRI, CT), serve per delineare strutture anatomiche o lesioni.

- **Classificazione**: una etichetta per immagine.
- **Segmentazione**: una mappa di etichette (mask) della stessa dimensione dell’immagine.
- Metriche tipiche:
  - **Dice coefficient** (sovrapposizione)
  - **IoU (Jaccard)** / MeanIoU
  - Precision, Sensitivity (Recall), Specificity (soprattutto in contesti clinici)

---

## 2) Dataset e obiettivi del progetto

### 2.1 Dataset: BraTS 2020 (MRI multimodale)
Il notebook lavora sul dataset **BraTS 2020** (Brain Tumor Segmentation), che contiene per ciascun paziente una risonanza magnetica in diverse modalità e una maschera di segmentazione:

- **Modalità MRI tipiche**: *FLAIR, T1, T1CE (T1 contrast-enhanced), T2*
- **Mask/seg**: etichette per regioni tumorali.

Nel notebook, il data loader utilizza **solo 2 canali**:
- **FLAIR**
- **T1CE**

Questa scelta riduce la dimensionalità dell’input ma può perdere informazione presente in T1 e T2.

### 2.2 Classi di segmentazione
Le classi (dopo conversione/uso nel notebook) sono:

- `0` = **Background / NOT tumor**
- `1` = **NECROTIC/CORE** (core/non-enhancing)
- `2` = **EDEMA**
- `3` = **ENHANCING**

Nel codice sono mappate in un dizionario `SEGMENT_CLASSES`.

### 2.3 Obiettivo
Allenare un modello di **semantic segmentation 2D** per predire, slice per slice, la maschera tumorale in 4 classi (background + 3 sotto-regioni tumorali), utilizzando come input le slice 2D delle MRI (FLAIR + T1CE).

---

## 3) Struttura del notebook e pipeline

### 3.1 Organizzazione dati
- Il dataset viene letto da una cartella (in origine su Google Drive in Colab).
- Si costruisce la lista degli ID (cartelle paziente) e poi si fa uno split:

- **Train/Validation/Test**:
  - prima split: `val` = 15%
  - seconda split: `test` = 15% del rimanente
  - il resto = `train`

(Lo split avviene tramite `train_test_split` di scikit-learn.)

### 3.2 Preprocessing
Dal notebook:

- **Selezione slice**:
  - `VOLUME_SLICES = 100`
  - `VOLUME_START_AT = 22`
  - quindi per ogni volume si prendono 100 slice a partire dalla slice 22 (tipicamente per evitare estremi “vuoti”).
- **Resize**:
  - immagini e maschere vengono portate a `IMG_SIZE = 128` (da 240×240 originali).
  - per la maschera viene usata interpolazione **nearest neighbor** (corretta per label discrete).
- **Normalizzazione**:
  - l’input `X` viene normalizzato dividendo per `max(X)` (con epsilon per stabilità).
- **One-hot encoding**:
  - la maschera (H×W) diventa (H×W×4) con `tf.one_hot(depth=4)`.

### 3.3 DataGenerator (Keras Sequence)
Il notebook implementa un `DataGenerator` che, per ogni batch:

1. carica i file NIfTI (`.nii`) di **FLAIR**, **T1CE** e **SEG**
2. estrae le slice selezionate
3. fa resize a 128×128
4. costruisce `X` con **2 canali** (FLAIR, T1CE)
5. costruisce `y` e poi la converte in one-hot (4 classi)

Questo permette di addestrare un modello 2D su slice pur partendo da volumi 3D.

---

## 4) Modello utilizzato: U-Net 2D

### 4.1 Perché U-Net
La **U-Net** è uno standard per segmentazione biomedicale perché combina:

- un **encoder** (downsampling) che cattura contesto globale
- un **decoder** (upsampling) che ricostruisce la mappa di segmentazione
- **skip connections** tra livelli corrispondenti (per recuperare dettagli spaziali)

### 4.2 Architettura nel notebook
Nel notebook la funzione `build_unet()` costruisce una U-Net con:

- Input: `(128, 128, 2)`
- Encoder con filtri crescenti: **32 → 64 → 128 → 256 → 512**
- Bottleneck con **Dropout(0.2)**
- Decoder con upsampling e concatenazione con le feature dell’encoder (skip connections)
- Output: `Conv2D(4, (1,1), activation='softmax')` per le **4 classi**

Loss: **categorical cross-entropy** (coerente con softmax + one-hot).

Ottimizzatore: **Adam** (`learning_rate = 0.001`).

---

## 5) Metriche e criteri di valutazione

### 5.1 Dice coefficient
Il notebook usa un **Dice medio sulle classi** (media su 4 classi), calcolando per ogni classe:

\[
Dice_c = \frac{2|P_c \cap G_c| + \epsilon}{|P_c| + |G_c| + \epsilon}
\]

e poi media sui c.

### 5.2 MeanIoU
Usa `tf.keras.metrics.MeanIoU(num_classes=4)` per misurare l’IoU medio.

### 5.3 Precision / Sensitivity / Specificity
Il notebook calcola anche:

- **Precision** = TP / (TP + FP)
- **Sensitivity (Recall)** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)

Sono presenti anche varianti “tumor-only” (escludendo il background), utili quando lo sbilanciamento tra background e tumore è forte.

### 5.4 Dice per classe
Oltre al Dice medio, il notebook calcola:
- Dice necrotic/core
- Dice edema
- Dice enhancing

---

## 6) Training

- Epoche: **40**
- `steps_per_epoch = len(training_generator)`
- Validation su `valid_generator`

Il notebook usa anche callback (riduzione LR, checkpoint, logging su CSV) e produce grafici di:
- Dice (train/val)
- Dice per classe (train/val)
- Loss CCE (train/val)
- Accuracy (train/val)

---

## 7) Risultati ottenuti (test set)

Dal notebook, valutazione sul **test set**:

- **Loss**: `0.0234`
- **Accuracy**: `0.9924`
- **MeanIoU**: `0.84`
- **Dice coefficient (medio)**: `0.6183`
- **Precision**: `0.9936`
- **Sensitivity**: `0.9916`
- **Specificity**: `0.9979`
- **Dice necrotic/core**: `0.5248`
- **Dice edema**: `0.6903`
- **Dice enhancing**: `0.6655`

### 7.1 Interpretazione
- **Accuracy molto alta (≈0.992)**: tipicamente in segmentazione medicale può essere gonfiata dal fatto che il **background domina** (molti pixel non tumorali). Quindi accuracy da sola non è sufficiente.
- **MeanIoU 0.84**: valore buono, indica una sovrapposizione media elevata.
- **Dice medio 0.618**: moderato; più informativo dell’accuracy perché sensibile alla sovrapposizione delle regioni.
- **Dice per classe**:
  - necrotic/core è la più difficile (≈0.525)
  - edema (≈0.690) e enhancing (≈0.666) sono migliori
  - questo è coerente con la pratica: alcune regioni sono più piccole/irregolari e più soggette a errori.

---

## 8) Limiti e possibili miglioramenti

1. **Input multimodale incompleto**: usare anche T1 e T2 (4 canali) spesso migliora le performance su BraTS.
2. **Approccio 2D slice-wise**: si perde coerenza 3D. Un 3D U-Net o un modello 2.5D (stack di slice) può migliorare.
3. **Loss**: provare combinazioni come `DiceLoss + CCE` oppure focal loss per gestire sbilanciamento.
4. **Data augmentation**: rotazioni, flip, elastic deformations (attenzione: coerenti su immagine e mask).
5. **Post-processing**: rimozione di piccole componenti connesse spurie, smoothing morfologico, ecc.
6. **Valutazione**: riportare anche metriche tumor-only in modo esplicito, e magari metriche per “whole tumor / tumor core / enhancing tumor” come nella convenzione BraTS.

---

## 9) Conclusione
Il notebook implementa una pipeline completa per segmentazione su BraTS 2020 con:
- data generator per NIfTI
- preprocessing (slice selection, resize, one-hot)
- modello **U-Net 2D** (input 2 canali)
- training e logging
- valutazione su test set con metriche clinicamente rilevanti

I risultati mostrano ottime metriche globali (accuracy/specificity/precision), mentre le metriche di sovrapposizione (Dice) evidenziano la difficoltà sulle classi tumorali più complesse, in particolare **necrotic/core**.
