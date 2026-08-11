# COUGHVID Audio Capture Firmware

This ESP32-C3 firmware prepares raw microphone recordings for the host-side COUGHVID pipeline. It configures a standard I2S mono input at 16 kHz and signed 16-bit samples. The firmware does not claim a trained model, diagnosis, or clinical accuracy.

## Hardware

The default board mapping uses the ESP32-C3-DevKitM-1 status LED on GPIO 8 and BOOT button on GPIO 9. The I2S microphone defaults are BCLK GPIO 4, WS GPIO 5, and DIN GPIO 6. Confirm these three connections against the actual microphone module before capture. GPIO 11–17 are reserved and must not be used.

## Capture contract

The intended capture window is four seconds of mono PCM. Raw recordings are not overwritten. Host-side preprocessing remains authoritative for normalization, candidate segmentation, and the restrained stationary spectral gate: a 250 ms noise estimate, 12 dB reduction, and 8% spectral floor. Change and validate these settings explicitly before training.

## Build

Use ESP-IDF v5.5 targeting ESP32-C3. Build with the ESP-IDF build tool, then flash and monitor the device. The current firmware validates the I2S backend and reports its configuration; PCM export, button control, and model inference are subsequent integration steps once the microphone wiring and model tensor contract are confirmed.
