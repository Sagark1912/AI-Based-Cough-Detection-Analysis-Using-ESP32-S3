#ifndef APP_CONFIG_H
#define APP_CONFIG_H

/* ESP32-C3-DevKitM-1 board-safe application configuration. */
#define APP_STATUS_LED_GPIO 8
#define APP_BUTTON_GPIO 9

/* Confirm these three connections against the microphone board before use. */
#define APP_I2S_BCLK_GPIO 4
#define APP_I2S_WS_GPIO 5
#define APP_I2S_DIN_GPIO 6

#define APP_SAMPLE_RATE_HZ 16000
#define APP_PCM_BITS 16
#define APP_CAPTURE_SECONDS 4
#define APP_CAPTURE_SAMPLES (APP_SAMPLE_RATE_HZ * APP_CAPTURE_SECONDS)
#define APP_I2S_DMA_FRAME_NUM 256
#define APP_I2S_DMA_DESC_NUM 4
#define APP_READ_CHUNK_BYTES 1024
#define APP_CAPTURE_QUEUE_LENGTH 8
#define APP_BUTTON_DEBOUNCE_MS 250
#define APP_NOISE_ESTIMATE_MS 250
#define APP_NOISE_REDUCTION_DB 12
#define APP_SPECTRAL_FLOOR_PERCENT 8

#endif
