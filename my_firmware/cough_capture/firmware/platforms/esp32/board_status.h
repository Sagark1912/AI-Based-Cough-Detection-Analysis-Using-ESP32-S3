#ifndef BOARD_STATUS_H
#define BOARD_STATUS_H

#include "esp_err.h"
#include <stdbool.h>

typedef enum {
    BOARD_STATUS_IDLE,
    BOARD_STATUS_CAPTURE,
    BOARD_STATUS_ERROR
} board_status_t;

esp_err_t board_status_init(void);
bool board_button_pressed(void);
void board_status_set(board_status_t status);

#endif
