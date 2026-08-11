#include "board_status.h"
#include "app_config.h"
#include "driver/gpio.h"
#include "led_strip.h"
#include "esp_check.h"

static led_strip_handle_t s_led;

esp_err_t board_status_init(void)
{
    gpio_config_t button_config = {
        .pin_bit_mask = 1ULL << APP_BUTTON_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&button_config), "board", "button config failed");

    const led_strip_config_t strip_config = {
        .strip_gpio_num = APP_STATUS_LED_GPIO,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags.invert_out = false,
    };
    const led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 64,
        .flags.with_dma = false,
    };
    ESP_RETURN_ON_ERROR(led_strip_new_rmt_device(&strip_config, &rmt_config, &s_led),
                       "board", "LED init failed");
    return led_strip_clear(s_led);
}

bool board_button_pressed(void)
{
    return gpio_get_level(APP_BUTTON_GPIO) == 0;
}

void board_status_set(board_status_t status)
{
    uint32_t red = 0, green = 0, blue = 0;
    if (status == BOARD_STATUS_CAPTURE) {
        blue = 24;
    } else if (status == BOARD_STATUS_ERROR) {
        red = 24;
    } else {
        green = 8;
    }
    led_strip_set_pixel(s_led, 0, red, green, blue);
    led_strip_refresh(s_led);
}
