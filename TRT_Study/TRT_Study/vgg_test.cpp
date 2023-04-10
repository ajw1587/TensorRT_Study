#include "logging.h"
#include "utils.h"

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

// This example captures all warning messages but ignores informational messages
// Engine 실행, Serialize, Deserialize 등을 수행할때 Log를 확인할 수 있다.
// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_logger.html#details
static Logger gLogger;

// input/output data shape
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
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


void createEngine(unsigned int batch_size, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
	std::cout << "------------ Model Build Start ------------" << std::endl;
	// A network definition for input to the builder.
	INetworkDefinition* network = builder->createNetworkV2(0U);

	// Load .wts File
	// Weights : [DataType type, const void* values, int64_t count]
	std::map<std::string, Weights> weightMap = loadWeights("../Weights/vgg11.wts");

	// Create Input Data
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
	assert(data);

    // Create Model Layers
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 3, 3 }, weightMap["features.0.weight"], weightMap["features.0.bias"]);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    assert(pool1);
    pool1->setStrideNd(DimsHW{ 2, 2 });

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{ 3, 3 }, weightMap["features.3.weight"], weightMap["features.3.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    pool1->setStrideNd(DimsHW{ 2, 2 });

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{ 3, 3 }, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{ 3, 3 }, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    pool1->setStrideNd(DimsHW{ 2, 2 });

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.11.weight"], weightMap["features.11.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.13.weight"], weightMap["features.13.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    pool1->setStrideNd(DimsHW{ 2, 2 });

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.16.weight"], weightMap["features.16.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{ 3, 3 }, weightMap["features.18.weight"], weightMap["features.18.bias"]);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    pool1->setStrideNd(DimsHW{ 2, 2 });

    pool1 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 1, 1 });
    pool1->setStride(DimsHW{ 1, 1 });

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 4096, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);
    relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(batch_size);   // 사용할 Batch Size 설정
    config->setMaxWorkspaceSize(1 << 20);   // 사용할 메모리 크기 설정

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    std::cout << "==== model build done ====" << std::endl << std::endl;

    std::cout << "==== model selialize start ====" << std::endl << std::endl;
    std::ofstream p(engineFileName, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl << std::endl;
    }
    p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    std::cout << "==== model selialize done ====" << std::endl << std::endl;

    engine->destroy();
    network->destroy();
    p.close();
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
}


void main()
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 1. 변수 선언
	unsigned int batch_size = 1;		// 사용할 배치 사이즈 값
	bool serialize = false;				// Engine 생성 유무, true : 엔진 생성
	char engineFileName[] = "vgg11";	// Engine 이름

	char engine_file_path[256];
	sprintf(engine_file_path, "../Engine/%s.engine", engineFileName);	// Engine 저장 경로
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 2. Engine 존재 유무 확인
	// true면 무조건 Engine 만들기
	// false면 없으면 만들기
	bool exist_engine = false;
	if ((access(engine_file_path, 0) != -1)) {
		exist_engine = true;
	}
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Engine 생성
	if (!((serialize == false)/*Serialize 강제화 값*/ && (exist_engine == true) /*vgg.engine 파일이 있는지 유무*/)) {
		std::cout << "------------ Create " << engineFileName << " Engine ------------" << std::endl;
		
		// Builds an engine from a network definition.
		IBuilder* builder = createInferBuilder(gLogger);

		// Holds properties for configuring a builder to produce an engine.
		IBuilderConfig* config = builder->createBuilderConfig();

        // Create Engine
        createEngine(batch_size, builder, config, DataType::kFLOAT, engine_file_path);

        // Destroy builder, config
        builder->destroy();
        config->destroy();
	}
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4. Engine file 로드 
    char* trtModelStream{ nullptr };// 저장된 스트림을 저장할 변수
    size_t size{ 0 };
    std::cout << "------------ Engine file load ------------" << std::endl << std::endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cout << "[ERROR] Engine file load error" << std::endl;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 5. Engine File 로드 후 Engine 생성
    std::cout << "------------ Engine file deserialize ------------" << std::endl << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;

    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // GPU에서 입력과 출력으로 사용할 메모리 공간할당
    CHECK(cudaMalloc(&buffers[inputIndex], batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
    CHECK(cudaMalloc(&buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float)));
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 6. 입력으로 사용할 이미지 준비하기
    std::string img_dir = "../VGG11_py/data/";
    std::vector<std::string> file_names;
    //if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
    //    std::cerr << "[ERROR] Data search error" << std::endl;
    //}
    //else {
    //    std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
    //}
    cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    cv::Mat ori_img;
    std::vector<uint8_t> input(batch_size * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
    std::vector<float> outputs(OUTPUT_SIZE);
    for (int idx = 0; idx < batch_size; idx++) { // mat -> vector<uint8_t> 
        cv::Mat ori_img = cv::imread(file_names[idx]);
        cv::resize(ori_img, img, img.size()); // input size로 리사이즈
        memcpy(input.data(), img.data, batch_size * INPUT_H * INPUT_W * INPUT_C);
    }
    std::cout << "===== input load done =====" << std::endl << std::endl;

    uint64_t dur_time = 0;
    uint64_t iter_count = 100;

    // CUDA 스트림 생성
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    //warm-up
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    context->enqueue(batch_size, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 7. Inference
    for (int i = 0; i < iter_count; i++) {
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        context->enqueue(batch_size, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
        std::cout << dur << " milliseconds" << std::endl;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 8. 결과 출력
    std::cout << "==================================================" << std::endl;
    std::cout << "===============" << engineFileName << "===============" << std::endl;
    std::cout << iter_count << " th Iteration" << std::endl;
    std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
    std::cout << "Avg duration time with data transfer : " << dur_time / iter_count << " [milliseconds]" << std::endl;
    std::cout << "FPS : " << 1000.f / (dur_time / iter_count) << " [frame/sec]" << std::endl;
    int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
    std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << std::endl;
    //std::cout << "Class Name : " << class_names[max_index] << std::endl;
    std::cout << "==================================================" << std::endl;

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    return;
}