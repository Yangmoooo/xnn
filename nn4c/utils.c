#include "utils.h"

#include <math.h>
#include <stdint.h>

void softmax(float* input, int32_t size)
{
    float max = input[0], sum = 0;
    for (int32_t i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int32_t i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int32_t i = 0; i < size; i++) input[i] /= sum;
}
