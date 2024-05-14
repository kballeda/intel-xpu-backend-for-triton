#include <typeinfo>
// STL includes
#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <iomanip> 
// local includes
#include "zello_init.h"
#include <level_zero/ze_api.h>
#include <cassert>
#include <complex>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>
#include <sycl/kernel_bundle.hpp>
#include <variant>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <boost/type_index.hpp>


#define check(ans)                                                             \
  { do_check((ans), __FILE__, __LINE__); }
void do_check(ze_result_t code, const char *file, int line) {
  if (code != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "Failed: %d at %s %d\n", code, file, line);
    exit(1);
  }
}

enum DataType{
  INTEGER,
  FLOAT,
  HALF,
  STRING,
  DOUBLE,
  LONG
};

class tritonReproducer {
    public:
        tritonReproducer() = default;
        ~tritonReproducer() = default;
        sycl::kernel createKernel();
        template<typename T> std::vector<T> file_ops(std::string filename);
        void printHelp();
        template<typename T>
        T* allocateDevBuffer(size_t size);
        template <typename T>
        T* setupBuffers(int vidx);
        template<typename T>
        void host2device(T *dev_buffer, T *host_buffer, size_t size);
        sycl::nd_range<3> gridConfig(uint32_t gridX,
                          uint32_t gridY,
                          uint32_t gridZ,
                          int num_warps,
                          int threads_per_warp);
        template<typename T>
        void device2host(T *buffer, T *host_buffer, size_t size);
        sycl::queue getQueue(void) {return tr_queue;}
        void readArguments(void);
        std::string spvFileName = "./spirv-bins/good.spv";
        std::vector<std::string> argInfo;
        void *dev_output;
        size_t host_output_size;
        enum DataType type;
        void *host_output;
    private:
        sycl::queue tr_queue;

};
