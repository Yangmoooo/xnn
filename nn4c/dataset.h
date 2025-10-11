#pragma once

#include <stdint.h>

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define IMAGE_SIZE 28

typedef struct {
    uint8_t *images, *labels;
    int32_t nImages;
} InputData;

void load_mnist_csv(const char* filename, uint8_t** images, uint8_t** labels, int32_t* nImages);

void shuffle_data(uint8_t* images, uint8_t* labels, int32_t n);
