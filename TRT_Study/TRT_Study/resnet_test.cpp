#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <map>
#include <io.h>     // access

#include "logging.h"
#include "NvInfer.h"

using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// 입력 이미지 사이즈 선언
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Logger 선언
static Logger gLogger;

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "------------ Loading weights : " << file << " ------------" << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];	// 16진수를 10진수로
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".weight"].count;
    
    float* gamma_var = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        gamma_var[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, gamma_var, len };

    float* beta_gamma_var = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        beta_gamma_var[i] = beta[i] - (gamma[i] * mean[i] / sqrt(var[i] + eps));
    }
    Weights shift{ DataType::kFLOAT, beta_gamma_var, len };

    float* p = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        p[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, p, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* batchnorm = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(batchnorm);
    return batchnorm;
}

IActivationLayer* BasicBlock(INetworkDefinition* network, std::map<std::string, Weights> weightMap, ITensor& input, int inc, int outc, int stride, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* cv01 = network->addConvolutionNd(input, outc, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(cv01);
    cv01->setStrideNd(DimsHW{ stride, stride });
    cv01->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *cv01->getOutput(0), lname + ".bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* cv02 = network->addConvolutionNd(*relu1->getOutput(0), outc, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    cv02->setPaddingNd(DimsHW{ 1, 1 });
    assert(cv02);
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *cv02->getOutput(0), lname + ".bn2", 1e-5);

    // 잔차 학습
    IElementWiseLayer* sum;
    if (inc == outc)
    {
        sum = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        IConvolutionLayer* cv03 = network->addConvolutionNd(input, outc, DimsHW(1, 1), weightMap[lname + ".downsample.0.weight"], emptywts);
        cv03->setStrideNd(DimsHW{ stride, stride });
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *cv03->getOutput(0), lname + ".downsample.1", 1e-5);
        sum = network->addElementWise(*bn2->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUB);
    }

    IActivationLayer* relu2 = network->addActivation(*sum->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    return relu2;
}


ICudaEngine* createEngine(unsigned int batch_size, IBuilder * builder, IBuilderConfig * config, DataType dt, char* engineFileName)
{
    std::cout << "------------ Model Build Start ------------" << std::endl;

    // Load Weights File
    std::map<std::string, Weights> weightMap = loadWeights("../Weights/resnet18.wts");
    //DataType type;      //!< The type of the weights.
    //const void* values; //!< The weight values, in a contiguous array.
    //int64_t count;      //!< The number of weights in the array.
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };   // bias 가 없는 Layer에 넣어주기

    // Make Network
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create Input Data
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
    assert(data);

    IConvolutionLayer* cv01 = network->addConvolutionNd(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
    assert(cv01);
    cv01->setStrideNd(DimsHW{ 2, 2 });
    cv01->setPaddingNd(DimsHW{ 3, 3 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *cv01->getOutput(0), "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* max_pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    max_pool1->setStrideNd(DimsHW{ 2, 2 });
    max_pool1->setPaddingNd(DimsHW{ 1, 1 });

    // layer1.0
    IActivationLayer* layer1_0_relu = BasicBlock(network, weightMap, *max_pool1->getOutput(0), 64, 64, 1, "layer1.0");
    // layer1.1
    IActivationLayer* layer1_1_relu = BasicBlock(network, weightMap, *layer1_0_relu->getOutput(0), 64, 64, 1, "layer1.1");

    // layer2.0
    IActivationLayer* layer2_0_relu = BasicBlock(network, weightMap, *layer1_1_relu->getOutput(0), 64, 128, 2, "layer2.0");
    // layer2.1
    IActivationLayer* layer2_1_relu = BasicBlock(network, weightMap, *layer2_0_relu->getOutput(0), 128, 128, 1, "layer2.1");

    // layer3.0
    IActivationLayer* layer3_0_relu = BasicBlock(network, weightMap, *layer2_1_relu->getOutput(0), 128, 256, 2, "layer3.0");
    // layer3.1
    IActivationLayer* layer3_1_relu = BasicBlock(network, weightMap, *layer3_0_relu->getOutput(0), 256, 256, 1, "layer3.1");

    // layer4.0
    IActivationLayer* layer4_0_relu = BasicBlock(network, weightMap, *layer3_1_relu->getOutput(0), 256, 512, 2, "layer4.0");
    // layer4.1
    IActivationLayer* layer4_1_relu = BasicBlock(network, weightMap, *layer4_0_relu->getOutput(0), 512, 512, 1, "layer4.1");

    // Adaptive average pooling
    IPoolingLayer* adap_avg_pool = network->addPoolingNd(*layer4_1_relu->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7, 7 });
    adap_avg_pool->setStrideNd(DimsHW{ 1, 1 });

    // fc layer : fc.weight, fc.bias
    IFullyConnectedLayer* fc = network->addFullyConnected(*adap_avg_pool->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);

    // final
    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name" << std::endl;
    network->markOutput(*fc->getOutput(0));

    // Build Engine
    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build finish" << std::endl;

    // destroy network
    network->destroy();

    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void doInference(IExecutionContext& context, float* input, std::vector<float> output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // input, output으로 사용할 메모리 공간잡기
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // TensorRT engine에 데이터 넣어주고 결과값 가져오기
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output.data(), buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main()
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 1. 변수 선언
    unsigned int batch_size = 1;
    const char* engineFileName = "resnet18";

    char engine_file_path[256];
    sprintf(engine_file_path, "../Engine/%s.engine", engineFileName);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 2. Engine 확인 -> true : 없으면 engine 만들기, false : 엔진 존재
    bool make_engine;
    if (access(engine_file_path, 0) == -1) {
        make_engine = true;
    }
    else
    {
        make_engine = false;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 3. engine 생성
    if (make_engine)
    {
        std::cout << "------------ Create " << engineFileName << " Engine ------------" << std::endl;

        // Make Builder
        IBuilder* builder = createInferBuilder(gLogger);

        // Make Config
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create Engine
        ICudaEngine* engine = createEngine(batch_size, builder, config, DataType::kFLOAT, engine_file_path);
        assert(engine != nullptr);

        IHostMemory* modelStream{ nullptr };
        modelStream = engine->serialize();

        // check engine file
        std::ofstream p("resnet18.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

        // Destroy builder, config
        modelStream->destroy();
        builder->destroy();
        config->destroy();
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4. engine 사용하기
    char* trtModelStream{ nullptr };
    size_t size(0);

    std::ifstream file("resnet18.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    // 추론 시작
    std::vector<float> output(OUTPUT_SIZE);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, output, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}