#include "app.h"
#include "app_config.h"
#include "audio_capture.h"
#include "board_status.h"
#include "logger.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <inttypes.h>

static const char *TAG = "app";

void app_start(void)
{
    ESP_LOGI(TAG, "COUGHVID capture firmware started");
    ESP_LOGI(TAG, "Configure I2S microphone: BCLK=%d WS=%d DIN=%d",
             APP_I2S_BCLK_GPIO, APP_I2S_WS_GPIO, APP_I2S_DIN_GPIO);
    ESP_LOGI(TAG, "Raw format: %d Hz, mono, signed 16-bit, %d-second window",
             APP_SAMPLE_RATE_HZ, APP_CAPTURE_SECONDS);

    esp_err_t err = board_status_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "board initialization failed: %s", esp_err_to_name(err));
        return;
    }
    board_status_set(BOARD_STATUS_IDLE);

    err = audio_capture_init();
    if (err != ESP_OK) {
        board_status_set(BOARD_STATUS_ERROR);
        ESP_LOGE(TAG, "audio capture initialization failed: %s", esp_err_to_name(err));
        return;
    }

    ESP_LOGI(TAG, "Capture backend is ready; inference remains host-side until model contract is integrated");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
