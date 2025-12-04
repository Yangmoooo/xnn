#pragma once

#include <stdint.h>

#define HIDDEN_SIZE 256
#define MOMENTUM 0.9f

typedef struct {
    float *weights, *biases, *weight_momentum, *bias_momentum;
    int32_t input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

void init_layer(Layer* layer, int32_t in_size, int32_t out_size);

void forward_relu(Layer* restrict layer, float* restrict input, float* restrict output);
void forward_linear(Layer* restrict layer, float* restrict input, float* restrict output);
void backward(Layer* restrict layer, float* restrict input, float* restrict output_grad, float* restrict input_grad,
              float lr);
