# AI-Based-Cough-Detection-Analysis-Using-ESP32-S3

> **TinyML-based cough sound analysis and respiratory-condition screening on an ESP32-S3**

## Overview

**TinyCough-Edge** is an embedded AI project designed to analyze cough sounds using an **ESP32-S3** and a machine-learning model optimized for edge inference.

The project uses the **COUGHVID dataset** for model development and follows a complete pipeline from dataset validation and audio preprocessing to model training, quantization, and embedded inference.

The primary objective is to investigate whether useful cough-acoustic information can be analyzed locally on resource-constrained embedded hardware without continuously transmitting raw audio to a cloud server.

The system is designed as a **screening/research prototype**, not as a clinically validated medical diagnostic device.

---

## Key Features

* COUGHVID metadata validation and cleaning
* Validation of all available metadata fields
* Exact duplicate-row removal
* UUID-based audio-file matching
* Masked multi-task learning targets
* Conservative audio preprocessing
* Stationary spectral noise reduction
* Energy-based cough candidate segmentation
* Audio feature extraction
* Multi-task machine-learning architecture
* Model evaluation using per-task metrics
* Model quantization for embedded deployment
* TensorFlow Lite / TensorFlow Lite Micro inference
* ESP32-S3 edge-AI implementation
* Local inference without requiring continuous cloud connectivity
* OLED/web-based result display can be integrated into the embedded system

---

# System Architecture

```text
                    COUGHVID DATASET
                           │
                           ▼
                ┌──────────────────────┐
                │ Dataset Preparation  │
                │ coughvid_prepare.py  │
                └──────────┬───────────┘
                           │
                           ▼
                  Cleaned Metadata CSV
                           │
                           ▼
                ┌──────────────────────┐
                │ Audio Preprocessing  │
                │ audio_preprocess.py  │
                └──────────┬───────────┘
                           │
                           ▼
                Processed Audio Samples
                           │
                           ▼
                ┌──────────────────────┐
                │ Feature Extraction   │
                │ Mel / MFCC Features  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Multi-Task ML Model  │
                └──────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          Cough Type   Abnormalities  Diagnosis
              │            │            │
              └────────────┼────────────┘
                           │
                    Severity / Status
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation     │
                └──────────┬───────────┘
                           │
                           ▼
                    INT8 Quantization
                           │
                           ▼
                     .tflite Model
                           │
                           ▼
                ┌──────────────────────┐
                │      ESP32-S3        │
                │   TinyML Inference   │
                └──────────┬───────────┘
                           │
                           ▼
                Embedded Prediction
```

---

# 1. Dataset

The project uses the **COUGHVID dataset** as the primary source of cough recordings and associated metadata.

The dataset contains cough recordings accompanied by metadata describing different characteristics and expert-derived labels.

The project does not assume that every recording contains a valid label. Missing, unknown, and tied labels are retained during dataset preparation and are appropriately masked during training.

---

# 2. Dataset Preparation

The dataset preparation stage is implemented using:

```text
scripts/coughvid_prepare.py
```

This stage intentionally does **not** perform audio processing.

It is responsible only for dataset validation, metadata cleaning, target extraction, and reporting.

### Responsibilities

* Read COUGHVID metadata
* Validate the expected metadata fields
* Remove exact duplicate rows
* Match metadata UUIDs with corresponding `.wav` recordings
* Extract machine-learning targets
* Generate masks for unavailable labels
* Produce a cleaned CSV
* Generate a JSON cleaning report

Example:

```bash
python scripts/coughvid_prepare.py \
  --metadata "metadata_compiled.csv" \
  --audio-root "coughvid_20211012" \
  --out artifacts/coughvid_clean.csv \
  --allow-recording-groups
```

---

# 3. Multi-Task Learning Targets

The project uses multiple prediction heads rather than treating cough analysis as a single classification problem.

## Cough Type

Possible classes:

```text
dry
wet
unknown
```

Unknown labels are retained for reporting but are masked during training.

---

## Abnormalities

The abnormalities target is a masked multi-label task.

Possible labels include:

```text
wheezing
dyspnea
congestion
nothing
```

A recording can potentially contain multiple abnormalities.

---

## Diagnosis

Possible diagnosis categories include:

```text
COVID-19
healthy_cough
lower_infection
upper_infection
obstructive_disease
```

Missing or unavailable diagnosis labels are masked during training.

---

## Severity

Possible severity classes:

```text
mild
pseudocough
severe
unknown
```

Unknown labels are retained for reporting but are masked during training.

---

## Overall Status

Possible classes:

```text
healthy
symptomatic
COVID-19
```

Missing status information is masked.

---

# 4. Important Dataset Limitation

The supplied COUGHVID metadata provides a **recording UUID**, but does not provide a participant identifier suitable for guaranteeing a patient-independent train/validation/test split.

Therefore, the project does not automatically claim that its evaluation is patient-independent.

The optional:

```text
--allow-recording-groups
```

argument permits creation of a recording-level split while explicitly recording this limitation.

This distinction is important because recordings from the same individual could otherwise potentially appear across different dataset partitions.

---

# 5. Audio Preprocessing

Audio preprocessing is performed separately from dataset preparation.

The preprocessing stage is responsible for preparing recordings for feature extraction and model development.

The processing pipeline can include:

```text
Raw WAV
   │
   ▼
Audio validation
   │
   ▼
Normalization
   │
   ▼
Noise reduction
   │
   ▼
Cough candidate detection
   │
   ▼
Segment extraction
   │
   ▼
Feature generation
```

Raw recordings are never overwritten.

---

# 6. Noise Reduction

The project uses a restrained stationary spectral-gating approach.

The initial noise estimate is obtained from approximately the first:

```text
250 ms
```

The configured reduction level is:

```text
12 dB
```

with a spectral floor of:

```text
8%
```

These values are intentionally conservative.

Aggressive noise suppression can remove acoustic characteristics that may contain useful information for cough classification.

Therefore, raw and processed audio should be compared during validation before using the preprocessing configuration for final model training.

---

# 7. Feature Extraction

After preprocessing, cough segments are converted into machine-learning features.

Possible feature representations include:

* Mel spectrogram
* MFCC
* Log-Mel spectrogram
* Other compact acoustic representations

A typical processing flow is:

```text
Audio waveform
       │
       ▼
Windowing
       │
       ▼
FFT
       │
       ▼
Mel filter bank
       │
       ▼
Log transformation
       │
       ▼
Feature tensor
```

The selected representation is then supplied to the neural network.

---

# 8. Multi-Task Model

Instead of training separate independent models for every task, the project can use a shared feature extractor with multiple output heads.

```text
                 Input Audio Features
                         │
                         ▼
                Shared Neural Network
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          Cough Type  Abnormality Diagnosis
              │          │          │
              └──────┬───┴──────┬───┘
                     │          │
                  Severity   Overall Status
```

The shared layers learn general acoustic representations while the individual heads specialize in different prediction tasks.

---

# 9. Training

The model is trained using the cleaned and preprocessed dataset.

Masked labels are excluded from the corresponding loss calculation.

Conceptually:

```text
Total Loss =
    Cough Loss
  + Abnormality Loss
  + Diagnosis Loss
  + Severity Loss
  + Status Loss
```

Only valid labels contribute to each task's loss.

This prevents missing expert labels from being incorrectly treated as negative or positive examples.

---

# 10. Model Evaluation

The project does not assume a particular accuracy before execution.

Evaluation should be performed using the actual held-out test data.

Metrics should be reported separately for each prediction head.

Recommended metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Per-class performance
* Prediction confidence

For the multi-label abnormality head, appropriate multi-label metrics should additionally be reported.

Example:

```text
Cough Type
├── Accuracy
├── Precision
├── Recall
└── F1-score

Abnormalities
├── Per-label precision
├── Per-label recall
└── F1-score

Diagnosis
├── Accuracy
├── Macro F1
└── Confusion Matrix

Severity
├── Accuracy
└── F1-score

Overall Status
├── Accuracy
└── F1-score
```

Actual values should be added only after training and evaluation.

---

# 11. Embedded Deployment

After training and validation, the model can be converted and quantized for deployment on an ESP32-S3.

```text
Trained Model
     │
     ▼
TensorFlow Lite Conversion
     │
     ▼
INT8 Quantization
     │
     ▼
.tflite
     │
     ▼
TensorFlow Lite Micro
     │
     ▼
ESP32-S3
```

The objective is to minimize:

* Flash usage
* RAM usage
* Inference time
* Model size
* Power consumption

while maintaining acceptable model performance.

---

# 12. Embedded Hardware

A representative hardware configuration is:

| Component           | Purpose                  |
| ------------------- | ------------------------ |
| ESP32-S3            | Embedded AI processor    |
| I2S MEMS microphone | Cough audio acquisition  |
| OLED display        | Local prediction display |
| Push button         | Start/stop recording     |
| Wi-Fi               | Optional connectivity    |
| Battery             | Portable operation       |

The exact hardware configuration can be changed depending on the final prototype.

---

# 13. Embedded Inference Pipeline

The final embedded operation is:

```text
User cough
    │
    ▼
I2S Microphone
    │
    ▼
ESP32-S3 Audio Buffer
    │
    ▼
Preprocessing
    │
    ▼
Feature Extraction
    │
    ▼
Quantized ML Model
    │
    ▼
Multi-Task Inference
    │
    ├── Cough Type
    ├── Abnormalities
    ├── Diagnosis Prediction
    ├── Severity
    └── Overall Status
    │
    ▼
OLED / Web Interface
```

The primary advantage of this architecture is that inference can be performed locally on the embedded device.

---

# 14. Repository Structure

A recommended repository structure is:

```text
tinycough-edge-esp32/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   └── README.md
│
├── scripts/
│   ├── coughvid_prepare.py
│   ├── audio_preprocess.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── quantize.py
│
├── artifacts/
│   ├── coughvid_clean.csv
│   └── reports/
│
├── models/
│   ├── trained_model/
│   ├── model.tflite
│   └── model_int8.tflite
│
├── embedded/
│   └── esp32s3/
│       ├── src/
│       ├── include/
│       └── README.md
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   └── model_evaluation.ipynb
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    └── deployment.md
```

Large datasets and raw COUGHVID recordings should **not** be committed to the GitHub repository.

---

# 15. Example Dataset Preparation Command

```bash
python scripts/coughvid_prepare.py \
  --metadata "metadata_compiled.csv" \
  --audio-root "coughvid_20211012" \
  --out artifacts/coughvid_clean.csv \
  --allow-recording-groups
```

The script produces:

```text
artifacts/
└── coughvid_clean.csv
```

along with the corresponding cleaning/report information.

---

# 16. Reproducibility

The project separates each stage so that the complete pipeline can be reproduced:

```text
Dataset
   ↓
Validation
   ↓
Cleaning
   ↓
Audio preprocessing
   ↓
Feature extraction
   ↓
Training
   ↓
Evaluation
   ↓
Quantization
   ↓
Embedded deployment
```

This separation also makes it possible to compare preprocessing configurations and determine whether noise reduction or segmentation improves model performance.

---

# 17. Safety and Scope

This project is an **embedded AI research and screening prototype**.

It should not be interpreted as a medical diagnostic system.

Predictions generated by the model represent statistical outputs learned from the training dataset and do not replace:

* Medical examination
* Laboratory testing
* Clinical diagnosis
* Professional medical advice

The project also inherits limitations from the COUGHVID dataset, including label availability, recording variability, dataset bias, and limitations in participant-level identification.

---

# 18. Current Development Status

The project is developed as a staged pipeline.

### Completed / implemented

* [x] COUGHVID metadata loading
* [x] Metadata validation
* [x] Duplicate-row removal
* [x] UUID/WAV matching
* [x] Target extraction
* [x] Missing-label masking
* [x] Dataset cleaning report
* [x] Conservative audio preprocessing design
* [x] Embedded AI architecture

### To be completed after execution

* [ ] Feature extraction
* [ ] Model training
* [ ] Validation
* [ ] Test evaluation
* [ ] Model quantization
* [ ] `.tflite` generation
* [ ] ESP32-S3 deployment
* [ ] Real-time microphone inference
* [ ] Embedded performance measurement

No model accuracy or `.tflite` artifact should be claimed until these stages have actually been executed.

---

# 19. Future Improvements

Potential future development includes:

* Improved cough-event detection
* More robust environmental-noise handling
* Lightweight CNN architecture
* Depthwise-separable convolutions
* INT8-aware training
* ESP32-S3 inference optimization
* Real-time inference
* Confidence thresholding
* Power optimization
* Web-based local dashboard
* BLE/Wi-Fi data transfer
* Additional audio datasets
* Participant-independent evaluation where suitable identifiers are available
* Hardware enclosure and battery-powered operation

---

# 20. Technologies Used

### Hardware

* ESP32-S3
* I2S MEMS microphone
* OLED display
* Optional battery and push-button interface

### Software

* Python
* NumPy
* Pandas
* Librosa / audio-processing libraries
* TensorFlow / Keras
* TensorFlow Lite
* TensorFlow Lite Micro
* ESP-IDF / Arduino framework, depending on implementation

### AI / Embedded Technologies

* Machine Learning
* Deep Learning
* Multi-Task Learning
* Digital Signal Processing
* Spectral Analysis
* Mel Spectrogram
* MFCC
* Model Quantization
* TinyML
* Edge AI

---

# 21. Project Objective

The main objective of TinyCough-Edge is to demonstrate a complete **audio-based Edge AI pipeline** in which cough recordings are transformed into compact acoustic features and analyzed using a resource-constrained embedded processor.

The project focuses on the engineering challenges involved in bringing an audio machine-learning workload from a public dataset to an embedded TinyML platform.

---

## Project Status

**Development Stage:** Embedded AI / TinyML Prototype

**Target Hardware:** ESP32-S3

**Dataset:** COUGHVID

**Inference:** Edge / Local

**Model:** Multi-task neural network

**Deployment Format:** TensorFlow Lite / TensorFlow Lite Micro

**Medical Status:** Research/screening prototype; not clinically validated

---

## Author

**Sagar K.**

Electronics & Communication Engineering
Embedded Systems | Edge AI | TinyML | IoT

---

## License

Add the license applicable to your source code and verify the COUGHVID dataset's own terms before redistributing any dataset files.

---

## Disclaimer

This repository contains research and engineering work for cough-sound analysis using machine learning. The outputs are experimental model predictions and must not be used as a substitute for professional medical diagnosis or treatment.

