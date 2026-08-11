#include "audio_capture.h"
#include "app_config.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"

static const char *TAG = "audio";
static i2s_chan_handle_t s_rx_channel;
static bool s_initialized;

esp_err_t audio_capture_init(void)
{
    if (s_initialized) {
        return ESP_OK;
    }

    const i2s_chan_config_t channel_config = {
        .id = I2S_NUM_AUTO,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = APP_I2S_DMA_DESC_NUM,
        .dma_frame_num = APP_I2S_DMA_FRAME_NUM,
        .auto_clear = false,
        .allow_pd = false,
        .intr_priority = 0,
    };
    ESP_RETURN_ON_ERROR(i2s_new_channel(&channel_config, NULL, &s_rx_channel), TAG,
                       "failed to allocate I2S RX channel");

    const i2s_std_config_t standard_config = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(APP_SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                         I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = APP_I2S_BCLK_GPIO,
            .ws = APP_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = APP_I2S_DIN_GPIO,
            .invert_flags = { 0 },
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx_channel, &standard_config), TAG,
                       "failed to initialize I2S standard mode");
    s_initialized = true;
    ESP_LOGI(TAG, "I2S capture configured: %d Hz mono %d-bit", APP_SAMPLE_RATE_HZ, APP_PCM_BITS);
    return ESP_OK;
}

esp_err_t audio_capture_start(void)
{
    ESP_RETURN_ON_ERROR(audio_capture_init(), TAG, "capture init failed");
    return i2s_channel_enable(s_rx_channel);
}

esp_err_t audio_capture_read(int16_t *samples, size_t sample_count, size_t *samples_read, uint32_t timeout_ms)
{
    if (!samples || !samples_read || !s_initialized) {
        return ESP_ERR_INVALID_ARG;
    }
    size_t bytes_read = 0;
    esp_err_t err = i2s_channel_read(s_rx_channel, samples, sample_count * sizeof(int16_t),
                                     &bytes_read, timeout_ms);
    *samples_read = bytes_read / sizeof(int16_t);
    return err;
}

esp_err_t audio_capture_stop(void)
{
    if (!s_initialized) {
        return ESP_OK;
    }
    return i2s_channel_disable(s_rx_channel);
}
