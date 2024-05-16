# Overview
The Triton reproducer is designed to facilitate the creation of reproducers for different Triton test case failures. It serves as a generic template for reproducer creation.


# Environment
    1. source /opt/intel/oneAPI/2024.1/setvars.sh 
    2. Set PATH variable pointing to Intel Level Zero headers

# Directories


data       - This directory contains all the binary files (.bin) consisting of input/output array binary data dumped from Triton backend 
             [Arrays: a, b, Torch_output, Triton_output]
src        - This directory contains all the source files 
spirv-bins - This directory contains SPIRV binary dumps from Triton, these files contains actual kernels in encrypted form.

# Reproducer Compilation
make            - By default it builds the tritonspvc++.exe

make VERBOSE=1  - Enables logging of Output Results

make DEBUG=1    - Enables debug mode


# Reproducer Usage:

./tritonspvc++.exe [By default it picks ./spirv-bins/good.spv and input.txt from current directory]
./tritonspvc++.exe --spv <path to SPIRV>
