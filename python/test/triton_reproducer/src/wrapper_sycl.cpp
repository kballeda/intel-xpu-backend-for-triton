#include "wrapper_sycl.hpp"

// File operations
template<typename fp>
std::vector<fp> tritonReproducer::file_ops(std::string filename) {
    // Create input data from binary file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << std::endl;
        exit(1);
    }

    // Read the file size to determine the number of elements
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate float16 elements
    size_t num_elements = filesize / sizeof(fp);
    std::vector<fp> data(num_elements);
    file.read(reinterpret_cast<char *>(data.data()), filesize);
    file.close();

    return data;
}

void tritonReproducer::printHelp() {
    std::cout << "Usage: gsd7949.exe [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help       Display this help message" << std::endl;
    std::cout << " -spv, --spv <good.spv/bad.spv> Default set to Good SPV file" << std::endl;
}

sycl::kernel tritonReproducer::createKernel() {
    ze_result_t status;
    const ze_device_type_t type = ZE_DEVICE_TYPE_GPU;
 
    ze_driver_handle_t pDriver = nullptr;
    ze_device_handle_t pDevice = nullptr;
     
    if ( init_ze()) {
        uint32_t driverCount = 0;
        status = zeDriverGet(&driverCount, nullptr);
        std::vector<ze_driver_handle_t> drivers(driverCount);
        status = zeDriverGet(&driverCount, drivers.data());
 
        for (uint32_t driver = 0; driver < driverCount; driver++) {
                pDriver = drivers[driver];
                pDevice = findDevice(pDriver, type);
                if (pDevice)
                   break;
        }
    }
 
    if (!pDevice) {
        std::cout << "Did not find matching " << to_string(type) << " device! " << "\n";
        exit(1);
    }
 
    // L0: Context creation
    ze_context_handle_t context;
    ze_context_desc_t context_desc = {};
    zeContextCreate(pDriver, &context_desc, &context);
 
    // L0: Queue creation
    ze_device_properties_t deviceProperties = {};
    check(zeDeviceGetProperties(pDevice, &deviceProperties));
 
    // Create command queue
    uint32_t numQueueGroups = 0;
    check(zeDeviceGetCommandQueueGroupProperties(pDevice, &numQueueGroups, nullptr));
    if (numQueueGroups == 0) {
        exit(1);
    }
    std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
    check(zeDeviceGetCommandQueueGroupProperties(pDevice, &numQueueGroups,
                                                        queueProperties.data()));
 
    ze_command_queue_handle_t command_queue;
    ze_command_queue_desc_t cmdQueueDesc = {};
 
    for (uint32_t i = 0; i < numQueueGroups; i++) {
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
        }
    }
    cmdQueueDesc.index = 0;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    check(zeCommandQueueCreate(context, pDevice, &cmdQueueDesc, &command_queue));

    ze_command_list_desc_t command_list_desc = {};
    ze_command_list_handle_t command_list;
    check(zeCommandListCreate(context, pDevice, &command_list_desc, &command_list));
    
    // Read spv file and create a buffer
    auto spvBuffer = file_ops<uint8_t>(spvFileName.c_str());

    // Create module
    ze_module_desc_t moduleDesc = {};
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.inputSize = spvBuffer.size();
    moduleDesc.pInputModule = spvBuffer.data();
    ze_module_build_log_handle_t buildlog;
    ze_module_handle_t module;
    auto result = zeModuleCreate(context, pDevice, &moduleDesc, &module, &buildlog);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeModuleCreate failed\n";
        size_t szLog = 0;
        check(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
        char *strLog = (char *)malloc(szLog);
        check(zeModuleBuildLogGetString(buildlog, &szLog, strLog));
        std::cerr << "L0 build module failed. Log: " << strLog << std::endl;
        free(strLog);
        check(zeModuleBuildLogDestroy(buildlog));
        exit(1);
    }

    // Query kernel names
    uint32_t kernelCount = 0;
    result = zeModuleGetKernelNames(module, &kernelCount, nullptr);
    if (result != ZE_RESULT_SUCCESS || kernelCount == 0) {
        std::cerr << "zeModuleGetKernelNames failed\n";
        exit(1);
    }

    std::vector<const char*> kernelNames(kernelCount);
    result = zeModuleGetKernelNames(module, &kernelCount, kernelNames.data());
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeModuleGetKernelNames failed\n";
        exit(1);
    }

    // Select a kernel
    const char* kernelName = kernelNames[0]; // Assuming the first kernel is selected
    std::cout << kernelName << std::endl;

    // Create kernel
    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.pKernelName = kernelName;
    kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDesc.pNext = nullptr;
    kernelDesc.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
    kernelDesc.pKernelName = kernelName;
    ze_kernel_handle_t kernel;
    result = zeKernelCreate(module, &kernelDesc, &kernel);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeKernelCreate failed\n";
        exit(1);
    }

    // L0 to SYCL handover
    auto device = sycl::device(sycl::gpu_selector_v);
    std::cout << "GPU Device Information:" << std::endl;
    std::cout << "Name: " << device.get_info<sycl::info::device::name>() << std::endl;
    sycl::queue queue(device);
    auto ctx = queue.get_context();
    auto mod = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);

    auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
                {mod, kernel, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);

    std::string kernel_name = fun.get_info<sycl::info::kernel::function_name>();
    std::cout << "KaliLog: kernel_name "  << kernel_name << "Expected Args " << fun.get_info<sycl::info::kernel::num_args>() << std::endl;
    tr_queue = queue;
    return fun;
}

template<typename T> 
T* tritonReproducer::allocateDevBuffer(size_t size) {
    auto buffer = (T*)sycl::malloc_device<T>(size, getQueue());
    return buffer;
}

template <typename T>
void tritonReproducer::host2device(T *dev_buffer, T *host_buffer, size_t size) {
    getQueue().memcpy(dev_buffer, host_buffer, size).wait();
}

template <typename T>
void tritonReproducer::device2host(T *dev_buffer, T *host_buffer, size_t size) {
    getQueue().memcpy(host_buffer, dev_buffer, size).wait();
}

sycl::nd_range<3> tritonReproducer::gridConfig(uint32_t gridX,
                                               uint32_t gridY,
                                               uint32_t gridZ,
                                               int num_warps,
                                               int threads_per_warp)
{
    size_t global_range_x = gridX*threads_per_warp*num_warps;
    size_t global_range_y = gridY;
    size_t global_range_z = gridZ;
    size_t local_range_x = num_warps*threads_per_warp;
    size_t local_range_y = 1;
    size_t local_range_z = 1;
    sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
    sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
    sycl::nd_range<3> parallel_work_size(global_range, local_range);
    return parallel_work_size;
}

void tritonReproducer::readArguments(void) {
    std::ifstream inputFile("input.txt"); // Open the input file
    if (!inputFile) { // Check if the file is opened successfully
        std::cerr << "Failed to open the file." << std::endl;
        exit(1);
    }

    std::string line;
    while (getline(inputFile, line)) { // Read each line from the file
        std::stringstream ss(line);
        std::string token;
        while (getline(ss, token, ',')) { // Tokenize the line using comma as the delimiter
            // Trim leading and trailing whitespaces from the token
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                std::string word = token.substr(start, end - start + 1);
                //std::cout << word << " "; // Print the word
                argInfo.push_back(word);
            }
        }
        //std::cout << std::endl; // Print a newline after processing each line
    }
    inputFile.close();
    //for (auto temp : argInfo)
    //    std::cout << temp << std::endl;
}

template <typename T>
T* tritonReproducer::setupBuffers(int vidx) {
    auto host_mem = this->file_ops<T>(this->argInfo[vidx + 2]);
    auto dev_mem = this->allocateDevBuffer<T>(host_mem.size());
    auto arrType = stoi(this->argInfo[vidx + 3]);
    auto typeStr = boost::typeindex::type_id_with_cvr<T>().pretty_name();
    //std::cout << typeStr << std::endl;
    // If this buffer is a output buffer 
    if (arrType) {
        this->dev_output = dev_mem;
        this->host_output_size = host_mem.size();
        auto typeStr = boost::typeindex::type_id_with_cvr<T>().pretty_name();
        if (typeStr == "float")
            this->type = FLOAT;
        else if (typeStr == "integer")
            this->type = INTEGER;
        else if (typeStr == "sycl::_V1::detail::half_impl::half")
            this->type = HALF;
        else if (typeStr == "double")
            this->type = DOUBLE;
        else if (typeStr == "long")
            this->type = LONG;
    } else {
        this->host2device<T>(dev_mem, host_mem.data(), sizeof(T) * host_mem.size());
    }
    return dev_mem;
}

int main(int argc, char **argv) {
    tritonReproducer tr;
    // Default set to good SPV file
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            tr.printHelp();
            return 0;
        }

        if (arg == "-spv" || arg == "--spv") {
            tr.spvFileName =  argv[i + 1];
            break;      
        }
    }
    tr.readArguments();

    std::cout << "Using spvFileName: " << tr.spvFileName << std::endl;
    auto kernel = tr.createKernel();

    uint32_t gridX = 0, gridY = 0, gridZ = 0;
    int num_warps = 0, threads_per_warp = 0;

    // Submit the imported kernel
    tr.getQueue().submit([&](sycl::handler &cgh) {
        int vidx = 0;
        int argCnt = 0;
        int narg = 0;
        while (vidx < tr.argInfo.size()) {
            if (tr.argInfo[vidx] == "Array") {
                std::cout << tr.argInfo[vidx] << std::endl;
                if (tr.argInfo[vidx + 1] == "float") {
                    auto dev_mem = tr.setupBuffers<float>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                    auto arrType = stoi(tr.argInfo[vidx + 3]);
                    // If this buffer is a output buffer 
                    if (arrType) {
                        tr.dev_output = dev_mem;
                        tr.host_output_size = 32768;
                        tr.type = FLOAT;
                    }
                } else if (tr.argInfo[vidx + 1] == "half") {
                    auto dev_mem = tr.setupBuffers<sycl::half>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } else if (tr.argInfo[vidx + 1] == "int") {
                    auto dev_mem = tr.setupBuffers<int>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } else if (tr.argInfo[vidx + 1] == "long") {
                    auto dev_mem = tr.setupBuffers<int>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } else if (tr.argInfo[vidx + 1] == "int64") {
                    auto dev_mem = tr.setupBuffers<int64_t>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } else if (tr.argInfo[vidx + 1] == "uint32") {
                    auto dev_mem = tr.setupBuffers<uint32_t>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } else if (tr.argInfo[vidx + 1] == "uint64") {
                    auto dev_mem = tr.setupBuffers<uint64_t>(vidx);
                    cgh.set_arg(narg++, dev_mem);
                } 
            }

            if (tr.argInfo[vidx] == "Var") {
                std::cout << tr.argInfo[vidx] << std::endl;
                if (tr.argInfo[vidx + 1] == "float") {
                    float arg = stof(tr.argInfo[vidx + 2]);
                    std::cout << arg << std::endl;
                    cgh.set_arg(narg++, arg);
                }
                if (tr.argInfo[vidx + 1] == "int") {
                    int arg = stoi(tr.argInfo[vidx + 2]);
                    std::cout << arg << std::endl;
                    cgh.set_arg(narg++, arg);
                }
            }

            if (tr.argInfo[vidx] == "SM") {
                using share_mem_t = sycl::local_accessor<int8_t, 1>;
                share_mem_t local_buffer = share_mem_t(stoi(tr.argInfo[vidx + 2]), cgh);
                cgh.set_arg(narg++, local_buffer);
            }

            if (tr.argInfo[vidx] == "GDIM") {
                gridX = stoi(tr.argInfo[vidx + 1]);
                gridY = stoi(tr.argInfo[vidx + 2]);
                gridZ = stoi(tr.argInfo[vidx + 3]);
                num_warps = stoi(tr.argInfo[vidx + 4]);
                threads_per_warp = stoi(tr.argInfo[vidx + 5]);
                std::cout << gridX << " " << gridY << " " << gridZ << " " << num_warps << " " << threads_per_warp << std::endl;
                vidx += 2;
            }
            vidx += 4;
        }
        auto parallel_work_size = tr.gridConfig(gridX, gridY, gridZ, num_warps, threads_per_warp);
        cgh.parallel_for(parallel_work_size, kernel);
    });
    tr.getQueue().wait();

    // Get output from the device memory.
    if (tr.type == FLOAT) {
        tr.host_output = (float*) malloc(tr.host_output_size * sizeof(float));
        auto devData = (float*)tr.dev_output;
        // D2H - Transfer of C
        tr.device2host<float>(devData, (float*)tr.host_output, tr.host_output_size * sizeof(float));
    }

    auto torch_output = tr.file_ops<float>("./data/th.bin");
    float *ptr = (float*)tr.host_output;
    int idx = 0;
    for (int i = 0; i < tr.host_output_size; i++) {
        if (ptr[i] == torch_output[i])
            idx++;
        else {
            std::cout << "Mismatch Occured at " << i << std::endl;
            break;
        }
    }
    std::cout << "total elements" << tr.host_output_size << " " << idx << std::endl;
}
