#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include "opencv2/opencv.hpp"

void preprocess_cu_0(float* output, unsigned char* input, int batchSize, int height, int width, int channel, cudaStream_t stream);

int main(int argc, char** argv)
{
    std::vector<uint8_t> input(1 * 224 * 256 * 3);
    float* output = 0;
    cv::Mat img(224, 256, 3);
    cv::Mat ori_img = cv::imread("./Lenna.png");

    cv::resize(ori_img, img, img.size());

    memcpy(input.data(), img.data, 1 * 224 * 256 * 3);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    preprocess_cu_0(output, &input[0], 1, 224, 256, 3, stream);
    cudaDeviceSynchronize();

    return 0;
}