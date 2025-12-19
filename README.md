# Brain Tumor Segmentation with CNN (U-Net) on BraTS 2020 — Report

## 1) Introduction: neural networks, CNNs, and segmentation

### 1.1 Neural Networks
An **artificial neural network** is a model composed of layers of computational units that apply non-linear transformations to data. In simplified form:

- **Input**: vector/tensor of features.
- **Layers**: linear transformations (weights) + **activation functions** (ReLU, sigmoid, etc.) to introduce non-linearity.
- **Output**: prediction (classification, regression, etc.).
- **Training**: a **loss function** is minimized through optimization (e.g. **Adam**) using **backpropagation**.

In deep learning, increasing the depth and capacity of the model makes it possible to capture complex patterns, provided that sufficient data and appropriate regularization are available.

### 1.2 Convolutional Neural Networks (CNN)
**CNNs** are networks specialized for structured data such as images. Instead of fully-connected connections, they use **convolutions**:

- A kernel (filter) slides over the image and produces feature maps.
- Convolutions exploit **locality** and **weight sharing**, reducing the number of parameters and improving generalization.
- Typically, the following blocks are alternated:
  - **Conv2D + ReLU**
  - **Pooling** (e.g. MaxPooling) to reduce resolution and increase the receptive field
  - (Optional) **Dropout / BatchNorm** for stability and regularization

### 1.3 Segmentation (semantic segmentation)
**Segmentation** consists of predicting **one class for each pixel (or voxel)**. In medical imaging (MRI, CT), it is used to delineate anatomical structures or lesions.

- **Classification**: one label per image.
- **Segmentation**: a label map (mask) with the same dimensions as the image.
- Typical metrics:
  - **Dice coefficient** (overlap)
  - **IoU (Jaccard)** / MeanIoU
  - Precision, Sensitivity (Recall), Specificity (especially in clinical contexts)

---

## 2) Dataset and project objectives

### 2.1 Dataset: BraTS 2020 (multimodal MRI)
The notebook works on the **BraTS 2020** (Brain Tumor Segmentation) dataset, which contains for each patient a magnetic resonance scan in multiple modalities and a segmentation mask:

- **Typical MRI modalities**: *FLAIR, T1, T1CE (T1 contrast-enhanced), T2*
- **Mask/seg**: labels for tumor regions.

In the notebook, the data loader uses **only 2 channels**:
- **FLAIR**
- **T1CE**

This choice reduces the input dimensionality but may lose information present in T1 and T2.

### 2.2 Segmentation classes
The classes (after conversion/usage in the notebook) are:

- `0` = **Background / NOT tumor**
- `1` = **NECROTIC/CORE** (core/non-enhancing)
- `2` = **EDEMA**
- `3` = **ENHANCING**

In the code they are mapped in a `SEGMENT_CLASSES` dictionary.

### 2.3 Objective
To train a **2D semantic segmentation** model to predict, slice by slice, the tumor mask into 4 classes (background + 3 tumor sub-regions), using as input the 2D MRI slices (FLAIR + T1CE).

---

## 3) Notebook structure and pipeline

### 3.1 Data organization
- The dataset is read from a folder (originally on Google Drive in Colab).
- A list of IDs (patient folders) is built and then split into:

- **Train/Validation/Test**:
  - first split: `val` = 15%
  - second split: `test` = 15% of the remaining data
  - the rest = `train`

(The split is performed using `train_test_split` from scikit-learn.)

### 3.2 Preprocessing
From the notebook:

- **Slice selection**:
  - `VOLUME_SLICES = 100`
  - `VOLUME_START_AT = 22`
  - therefore, for each volume, 100 slices are taken starting from slice 22 (typically to avoid “empty” extremes).
- **Resize**:
  - images and masks are resized to `IMG_SIZE = 128` (from the original 240×240).
  - for the mask, **nearest neighbor** interpolation is used (correct for discrete labels).
- **Normalization**:
  - the input `X` is normalized by dividing by `max(X)` (with epsilon for stability).
- **One-hot encoding**:
  - the mask (H×W) becomes (H×W×4) using `tf.one_hot(depth=4)`.

### 3.3 DataGenerator (Keras Sequence)
The notebook implements a `DataGenerator` that, for each batch:

1. loads the NIfTI (`.nii`) files for **FLAIR**, **T1CE**, and **SEG**
2. extracts the selected slices
3. resizes them to 128×128
4. builds `X` with **2 channels** (FLAIR, T1CE)
5. builds `y` and then converts it to one-hot encoding (4 classes)

This allows training a 2D model on slices while starting from 3D volumes.

---

## 4) Model used: 2D U-Net

### 4.1 Why U-Net
**U-Net** is a standard for biomedical segmentation because it combines:

- an **encoder** (downsampling) that captures global context
- a **decoder** (upsampling) that reconstructs the segmentation map
- **skip connections** between corresponding levels (to recover spatial details)

### 4.2 Architecture in the notebook
In the notebook, the `build_unet()` function builds a U-Net with:

- Input: `(128, 128, 2)`
- Encoder with increasing filters: **32 → 64 → 128 → 256 → 512**
- Bottleneck with **Dropout(0.2)**
- Decoder with upsampling and concatenation with encoder features (skip connections)
- Output: `Conv2D(4, (1,1), activation='softmax')` for the **4 classes**

Loss: **categorical cross-entropy** (consistent with softmax + one-hot encoding).

Optimizer: **Adam** (`learning_rate = 0.001`).

---

## 5) Metrics and evaluation criteria

### 5.1 Dice coefficient
The notebook uses a **mean Dice over classes** (average over 4 classes), computing for each class:

\[
Dice_c = \frac{2|P_c \cap G_c| + \epsilon}{|P_c| + |G_c| + \epsilon}
\]

and then averaging over classes.

### 5.2 MeanIoU
`tf.keras.metrics.MeanIoU(num_classes=4)` is used to measure the mean IoU.

### 5.3 Precision / Sensitivity / Specificity
The notebook also computes:

- **Precision** = TP / (TP + FP)
- **Sensitivity (Recall)** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)

There are also “tumor-only” variants (excluding background), useful when class imbalance between background and tumor is strong.

### 5.4 Dice per class
In addition to the mean Dice, the notebook computes:
- Dice necrotic/core
- Dice edema
- Dice enhancing

---

## 6) Training

- Epochs: **40**
- `steps_per_epoch = len(training_generator)`
- Validation on `valid_generator`

The notebook also uses callbacks (LR reduction, checkpointing, CSV logging) and produces plots of:
- Dice (train/val)
- Dice per class (train/val)
- CCE Loss (train/val)
- Accuracy (train/val)

---

## 7) Results obtained (test set)

From the notebook, evaluation on the **test set**:

- **Loss**: `0.0234`
- **Accuracy**: `0.9924`
- **MeanIoU**: `0.84`
- **Mean Dice coefficient**: `0.6183`
- **Precision**: `0.9936`
- **Sensitivity**: `0.9916`
- **Specificity**: `0.9979`
- **Dice necrotic/core**: `0.5248`
- **Dice edema**: `0.6903`
- **Dice enhancing**: `0.6655`

### 7.1 Interpretation
- **Very high accuracy (≈0.992)**: in medical segmentation this can be inflated because the **background dominates** (many non-tumor pixels). Therefore, accuracy alone is not sufficient.
- **MeanIoU 0.84**: a good value, indicating high average overlap.
- **Mean Dice 0.618**: moderate; more informative than accuracy because it is sensitive to region overlap.
- **Dice per class**:
  - necrotic/core is the most difficult (≈0.525)
  - edema (≈0.690) and enhancing (≈0.666) perform better
  - this is consistent with practice: some regions are smaller/irregular and more error-prone.

---

## 8) Limitations and possible improvements

1. **Incomplete multimodal input**: using T1 and T2 as well (4 channels) often improves BraTS performance.
2. **2D slice-wise approach**: 3D consistency is lost. A 3D U-Net or a 2.5D model (slice stacks) could improve results.
3. **Loss function**: try combinations such as `DiceLoss + CCE` or focal loss to handle class imbalance.
4. **Data augmentation**: rotations, flips, elastic deformations (warning: must be consistent on both image and mask).
5. **Post-processing**: removal of small spurious connected components, morphological smoothing, etc.
6. **Evaluation**: explicitly report tumor-only metrics, and possibly metrics for “whole tumor / tumor core / enhancing tumor” as in the BraTS convention.

---

## 9) Conclusion
The notebook implements a complete segmentation pipeline on BraTS 2020 with:
- NIfTI data generator
- preprocessing (slice selection, resize, one-hot encoding)
- **2D U-Net** model (2-channel input)
- training and logging
- evaluation on the test set with clinically relevant metrics

The results show excellent global metrics (accuracy/specificity/precision), while overlap-based metrics (Dice) highlight the difficulty on more complex tumor classes, in particular **necrotic/core**.

