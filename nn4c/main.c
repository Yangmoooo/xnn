#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "model.h"
#include "utils.h"

// --- Hyperparameters ---
#define LEARNING_RATE 0.0005f
#define EPOCHS 20
#define TRAIN_SPLIT 0.8
#define TRAIN_CSV_PATH "./data/fashion-mnist/fashion-mnist_train.csv"
#define TEST_CSV_PATH "./data/fashion-mnist/fashion-mnist_test.csv"

float* train(Network* net, float* input, int32_t label, float lr)
{
    static float final_output[OUTPUT_SIZE];
    float hidden_output[HIDDEN_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    forward_relu(&net->hidden, input, hidden_output);
    forward_linear(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    for (int32_t i = 0; i < OUTPUT_SIZE; i++) {
        output_grad[i] = final_output[i] - (i == label);
    }

    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    for (int32_t i = 0; i < HIDDEN_SIZE; i++) {
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;
    }

    backward(&net->hidden, input, hidden_grad, NULL, lr);

    return final_output;
}

int32_t predict(Network* net, float* input)
{
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
    forward_relu(&net->hidden, input, hidden_output);
    forward_linear(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int32_t max_index = 0;
    for (int32_t i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > final_output[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

float evaluate(Network* net, InputData* data)
{
    if (data->nImages == 0) {
        return 0.0f;
    }

    int32_t correct = 0;
    float img_flat[INPUT_SIZE];
    for (int i = 0; i < data->nImages; i++) {
        uint8_t* current_image = data->images + i * INPUT_SIZE;
        for (int k = 0; k < INPUT_SIZE; k++) {
            img_flat[k] = current_image[k] / 255.0f;
        }
        if (predict(net, img_flat) == data->labels[i]) {
            correct++;
        }
    }
    return (float)correct / data->nImages;
}

int main(void)
{
    srand(time(NULL));

    // --- Data Loading ---
    InputData train_data = {0}, test_data = {0};
    printf("Loading training data from %s...\n", TRAIN_CSV_PATH);
    load_mnist_csv(TRAIN_CSV_PATH, &train_data.images, &train_data.labels, &train_data.nImages);
    printf("Loading testing data from %s...\n", TEST_CSV_PATH);
    load_mnist_csv(TEST_CSV_PATH, &test_data.images, &test_data.labels, &test_data.nImages);

    printf("Shuffling training data...\n");
    shuffle_data(train_data.images, train_data.labels, train_data.nImages);

    int32_t train_size = (int32_t)(train_data.nImages * TRAIN_SPLIT);
    int32_t validation_size = train_data.nImages - train_size;
    printf("Training on %d samples, validating on %d samples.\n", train_size, validation_size);

    printf("Initializing network...\n");
    Network network;
    init_layer(&network.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&network.output, HIDDEN_SIZE, OUTPUT_SIZE);

    float img_flat[INPUT_SIZE];

    for (int32_t epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t start = clock();
        float total_loss = 0;

        for (int32_t i = 0; i < train_size; i++) {
            uint8_t* current_image = train_data.images + i * INPUT_SIZE;
            for (int k = 0; k < INPUT_SIZE; k++) {
                img_flat[k] = current_image[k] / 255.0f;
            }

            float* final_output = train(&network, img_flat, train_data.labels[i], LEARNING_RATE);
            total_loss += -logf(final_output[train_data.labels[i]] + 1e-10f);
        }

        InputData validation_data = {
            .images = train_data.images + train_size * INPUT_SIZE,
            .labels = train_data.labels + train_size,
            .nImages = validation_size,
        };
        float val_accuracy = evaluate(&network, &validation_data);

        clock_t end = clock();
        double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Epoch %2d, Accuracy: %6.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", epoch + 1,
               val_accuracy * 100, total_loss / train_size, cpu_time_used);
    }

    printf("Starting final evaluation on the test set...\n");
    float test_accuracy = evaluate(&network, &test_data);
    printf("Final Test Accuracy: %.2f%%\n", test_accuracy * 100);

    printf("Cleaning up...\n");
    free(network.hidden.weights);
    free(network.hidden.biases);
    free(network.hidden.weight_momentum);
    free(network.hidden.bias_momentum);
    free(network.output.weights);
    free(network.output.biases);
    free(network.output.weight_momentum);
    free(network.output.bias_momentum);
    free(train_data.images);
    free(train_data.labels);
    free(test_data.images);
    free(test_data.labels);

    return 0;
}
