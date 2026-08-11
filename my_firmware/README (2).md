# Embedded Edge-AI Respiratory Screening System

ESP32-S3 firmware and TinyML research pipeline for offline respiratory screening. The system combines cough/breathing audio from an INMP441 with independent SpO₂/heart-rate and temperature sensors, then presents local screening information through an OLED, buzzer, and Wi-Fi dashboard.

> **Safety and scope:** This is a screening-assistance research system, not a clinical diagnostic device. It must abstain when data quality or confidence is insufficient. No model is approved for deployment until the evaluation gates in this document pass.

## Repository map

| Area | Purpose |
|---|---|
| `firmware/` | ESP-IDF application and platform code |
| `firmware/configs/app_config.h` | GPIOs and application constants |
| `firmware/app/` | Vendor-neutral startup/orchestration |
| `firmware/platforms/esp32/` | ESP-IDF I²C, I²S, Wi-Fi, HTTP, and board code |
| `scripts/` | Dataset cleaning, audio processing, features, training, evaluation, and shard tools |
| `artifacts/` | Generated datasets, features, models, reports, and checkpoints; not source code |
| `firmware/docs/` | Doxygen landing page and generated API documentation |

## Hardware wiring

| Device | ESP32-S3 pin |
|---|---|
| MAX30102 SDA/SCL | GPIO 8 / GPIO 9 |
| OLED SDA/SCL | GPIO 8 / GPIO 9 |
| INMP441 BCLK/WS/DOUT | GPIO 5 / GPIO 6 / GPIO 7 |
| DS18B20 DATA | GPIO 4 |
| Buzzer control | GPIO 15 |

Use 3.3 V and common ground. GPIO15 is only a control signal. Do not connect 3.3 V directly to GND; use a suitable buzzer module or transistor/MOSFET driver. The DS18B20 data line requires the appropriate external pull-up.

## Dataset and target contract

COUGHVID provides audio and metadata only. It does not provide synchronized SpO₂, heart rate, or temperature labels. Those are independent physical inputs at runtime.

The cleaned target heads are:

- **Cough type:** dry / wet; unknown labels are retained for reporting and masked from loss.
- **Abnormalities:** wheezing / dyspnea / congestion / nothing; represented as masked multi-label targets.
- **Diagnosis:** COVID-19 / healthy_cough / lower_infection / upper_infection / obstructive_disease.
- **Severity:** mild / pseudocough / severe; unknown labels are masked.
- **Overall status:** healthy / symptomatic / COVID-19; independently supervised, not derived from diagnosis.

Missing, tied, and unsupported annotations are never silently converted to negative labels. COUGHVID has recording UUIDs but no participant identifier in the supplied metadata, so participant-level leakage cannot be ruled out; reported metrics must disclose this limitation.

## Reproducible pipeline

```text
COUGHVID
  → validate all 52 metadata fields
  → clean metadata and preserve masks
  → quality filtering
  → cough detection and segmentation
  → conservative audio normalization/noise reduction
  → 128×128 time-frequency log-mel features
  → train-only augmentation and class balancing
  → masked multi-task CNN/CNN-GRU training
  → test metrics and calibration/abstention
  → quantization and resource measurement
  → TensorFlow Lite Micro integration
  → ESP32-S3 decision layer with independent sensors
  → OLED/buzzer/dashboard screening recommendation
```

Raw recordings are never overwritten. The current denoising implementation is restrained stationary spectral gating; it must be validated because aggressive denoising can remove clinically useful acoustic information.

## Current generated artifacts

- `artifacts/coughvid_clean_masked.csv` — complete metadata cleaning with training masks.
- `artifacts/audio_preprocessing_all_34434.csv` — all-row preprocessing manifest.
- `artifacts/features_timefreq_all_34434.complete.json` — verified 34,434-record time-frequency artifact.
- `artifacts/timefreq_shards_complete/` — verified 34 NumPy shards covering rows 0–34,433; 30,738 valid feature records.
- `artifacts/model_2dcnn_quick/test_evaluation.json` — detailed baseline test evaluation. This model is **not deployable**.

The current CNN baseline remains inadequate: diagnosis balanced accuracy is at the five-class random baseline, and important minority recalls remain zero. No `.tflite` model has been approved or deployed.

## Dataset preparation

From the project root:

```bat
python -m pip install -r scripts\requirements.txt
python scripts\coughvid_prepare.py --metadata "C:\Users\chait\OneDrive\Desktop\dataset\metadata_compiled.csv" --audio-root "C:\Users\chait\OneDrive\Desktop\dataset\covid dataset\public_dataset_v3\coughvid_20211012" --out artifacts\coughvid_clean_masked.csv --allow-recording-groups
```

## Firmware build and dashboard

Use ESP-IDF for `esp32s3`:

```bat
idf.py build
idf.py flash monitor
```

After the serial log reports an IP address, open `http://<device-ip>/` on the same LAN. The dashboard currently reports I²C discovery, audio readiness, placeholder vital-sign state, and model availability. The MAX30102/DS18B20 measurement drivers and TinyML runtime remain staged work.

## Deployment gates

Do not quantize or flash a TinyML model until all of the following are true:

1. Valid test-set metrics include balanced accuracy, macro-F1, per-class precision/recall, confusion matrices, and confidence calibration.
2. COVID-19, symptomatic, obstructive-disease, severe, and wet-cough recall are explicitly reviewed.
3. Low-confidence and insufficient-data abstention behavior is defined.
4. Evaluation uses unseen audio and documents the missing participant-ID limitation.
5. Quantized-versus-float accuracy, tensor-arena size, latency, RAM/PSRAM, and flash usage are measured.
6. Hardware-in-the-loop checks pass for audio, sensors, dashboard, OLED, and buzzer safety.
