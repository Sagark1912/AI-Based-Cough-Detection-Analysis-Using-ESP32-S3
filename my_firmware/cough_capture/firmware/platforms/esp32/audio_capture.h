#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

esp_err_t audio_capture_init(void);
esp_err_t audio_capture_start(void);
esp_err_t audio_capture_read(int16_t *samples, size_t sample_count, size_t *samples_read, uint32_t timeout_ms);
esp_err_t audio_capture_stop(void);

#endif
