#include "dataset.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_mnist_csv(const char* filename, uint8_t** images, uint8_t** labels, int32_t* nImages)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening CSV file");
        exit(1);
    }

    // 1. 先计算文件行数，以确定需要分配多少内存
    int32_t count = 0;
    char line_buffer[4096];  // 假设一行不超过 4096 字符
    while (fgets(line_buffer, sizeof(line_buffer), file)) count++;
    *nImages = count;
    printf("Found %d images in CSV file.\n", *nImages);
    rewind(file);  // 回到文件开头

    // 2. 分配内存，由于这里要修改 images 和 labels 指针，所以参数里使用了二级指针
    *images = malloc((*nImages) * INPUT_SIZE * sizeof(uint8_t));
    *labels = malloc((*nImages) * sizeof(uint8_t));
    if (!*images || !*labels) {
        fprintf(stderr, "Failed to allocate memory for dataset.\n");
        exit(1);
    }

    // 3. 逐行读取并解析数据
    for (int32_t i = 0; i < *nImages; i++) {
        if (fgets(line_buffer, sizeof(line_buffer), file) == NULL) {
            fprintf(stderr, "Error reading line %d from CSV.\n", i + 1);
            exit(1);
        }

        // 解析标签
        char* token = strtok(line_buffer, ",");
        (*labels)[i] = (uint8_t)atoi(token);

        // 解析 784 个像素值
        for (int32_t j = 0; j < INPUT_SIZE; j++) {
            token = strtok(NULL, ",");
            (*images)[i * INPUT_SIZE + j] = (uint8_t)atoi(token);
        }
    }
    fclose(file);
}

void shuffle_data(uint8_t* images, uint8_t* labels, int32_t n)
{
    uint8_t temp_image[INPUT_SIZE];

    for (int32_t i = n - 1; i > 0; i--) {
        int32_t j = rand() % (i + 1);

        if (i == j) continue;

        memcpy(temp_image, images + i * INPUT_SIZE, INPUT_SIZE);
        memcpy(images + i * INPUT_SIZE, images + j * INPUT_SIZE, INPUT_SIZE);
        memcpy(images + j * INPUT_SIZE, temp_image, INPUT_SIZE);

        uint8_t temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}
