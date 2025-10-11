#include "model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void init_layer(Layer* layer, int32_t in_size, int32_t out_size)
{
    int32_t n = in_size * out_size;
    // He 初始化，适合 ReLU
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(out_size, sizeof(float));

    // 生成 0.0 到 1.0 间的随机 float，然后转化到 -1.0 和 1.0 之间，最后乘上缩放因子
    for (int32_t i = 0; i < n; i++) {
        layer->weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * scale;
    }
}

void forward_relu(Layer* layer, float* input, float* output)
{
    for (int32_t i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
    }

    for (int32_t j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float* weight_row = &layer->weights[j * layer->output_size];
        for (int32_t i = 0; i < layer->output_size; i++) {
            output[i] += in_j * weight_row[i];
        }
    }

    // ReLU
    for (int32_t i = 0; i < layer->output_size; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
}

void forward_linear(Layer* layer, float* input, float* output)
{
    for (int32_t i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
    }

    for (int32_t j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float* weight_row = &layer->weights[j * layer->output_size];
        for (int32_t i = 0; i < layer->output_size; i++) {
            output[i] += in_j * weight_row[i];
        }
    }
}

void backward(Layer* layer, float* input, float* output_grad, float* input_grad, float lr)
{
    if (input_grad) {
        for (int32_t j = 0; j < layer->input_size; j++) {
            input_grad[j] = 0.0f;
            float* weight_row = &layer->weights[j * layer->output_size];
            for (int32_t i = 0; i < layer->output_size; i++) {
                input_grad[j] += output_grad[i] * weight_row[i];
            }
        }
    }

    for (int32_t j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        if (in_j == 0) continue;
        float* weight_row = &layer->weights[j * layer->output_size];
        float* momentum_row = &layer->weight_momentum[j * layer->output_size];
        for (int32_t i = 0; i < layer->output_size; i++) {
            float grad = output_grad[i] * in_j;
            momentum_row[i] = MOMENTUM * momentum_row[i] + lr * grad;
            weight_row[i] -= momentum_row[i];
        }
    }

    for (int32_t i = 0; i < layer->output_size; i++) {
        layer->bias_momentum[i] = MOMENTUM * layer->bias_momentum[i] + lr * output_grad[i];
        layer->biases[i] -= layer->bias_momentum[i];
    }
}
