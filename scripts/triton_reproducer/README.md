# Overview
The Triton reproducer is designed to facilitate the creation of reproducers for different Triton test case failures. It serves as a generic template for reproducer creation.


# Environment
    1. Setup oneAPI environment 
    2. Set PATH variable pointing to Intel Level Zero headers
    3. git apply scripts/triton_reproducer/tt_reproducer.patch

# Usage
Automated generic reproducer creation is supported for only test_matmul.py case at present.

    1. Update python/test/unit/operators/test_matmul.py to run a single case that requires reproducer
    2. ./scripts/triton_reproducer/tt_reproducer.sh "python3 -m pytest --verbose --device xpu python/test/unit/operators/test_matmul.py" 

Post running the step-2 above, generic reproducer output looks like below,

```
/intel-xpu-backend-for-triton$ ./scripts/triton_reproducer/tt_reproducer.sh "python3 -m pytest --verbose --device xpu python/test/unit/operators/test_matmul.py" 
==================================================== test session starts ====================================================
platform linux -- Python 3.10.12, pytest-7.4.3, pluggy-1.3.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /intel-xpu-backend-for-triton/python
plugins: rerunfailures-13.0, select-0.1.2, xdist-3.5.0
collected 1 item                                                                                                            

python/test/unit/operators/test_matmul.py::test_op[128-256-32-1-8-2-None-None-None-False-False-float16-float16-None-True-None-float16] PASSED [100%]

==================================================== 1 passed in 28.35s =====================================================
Processing file: /intel-xpu-backend-for-triton/tt_cache/9ab33bb9cd0339e3362384cf0d21b81201c37e699150f51f701f5562a9b85521/_kernel.spv
icpx -std=c++20 -fsycl -lpthread -lm -ldl -lze_loader /intel-xpu-backend-for-triton/scripts/triton_reproducer/src/wrapper_sycl.cpp -o tritonspvc++.exe
INPATH: /intel-xpu-backend-for-triton/scripts/triton_reproducer/
Using spvFileName: /intel-xpu-backend-for-triton/scripts/triton_reproducer//data/kernel.spv
Driver initialized.
Found ZE_DEVICE_TYPE_GPU device...
Driver version: 17002026
API version: 1.3
_kernel
GPU Device Information:
Name: Intel(R) Data Center GPU Max 1550
KaliLog: kernel_name _kernelExpected Args 10
 Total Elements = 32768 Total Matched Elements = 32768/32768
```

Share scripts/triton_reproducer directory with IGC Team and they need to run following command to reproduce at their end.
```
./tritonspvc++.exe -spv ./data/kernel.spv
```

# Future Work
  1. Data type support may be limited and needs to be expanded.
  2. Support dot and other test cases on priority
