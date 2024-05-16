# Overview
The Triton reproducer is designed to facilitate the creation of reproducers for different Triton test case failures. It serves as a generic template for reproducer creation.


# Environment
    1. Setup oneAPI environment 
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

# Input Format
This reproducer template program accepts an input file that includes input arguments and their format for a SPIRV kernel.

| Arg Type | Data Type | Input | Type of Buffer |
|----------|-----------|-------|----------------|
| ARRAY    | half      | ./data/a.bin | 0       | 
| ARRAY    | half      | ./data/b.bin | 0       |
| ARRAY    | float     | ./data/torch_output.bin | 1 |
| VAR      | int       | 128 | 0 |
| VAR      | int       | 256 | 0 |
| VAR      | int       | 32 | 0 |
| VAR      | int       | 32 | 0 |
| VAR      | int       | 256 | 0 |
| VAR      | int       | 256 | 0 |
| SM       | int       | 32768 | 0 |
| GDIM     | 1 | 1 | 1 | 8 | 32 |

The table provided above describes the format of the input that the reproducer takes in order to derive the kernel input arguments in a generic manner.

**ARRAY** - Input/Output buffers

**VAR**   - Scalar variable

**SM**    - Shared Memory Size

**GDIM**  - Global Grid Dimensions (GridX, GridY, GridZ, num_warps and Threads Per Warp)

Example

# Reproducer Usage:

./tritonspvc++.exe [By default it picks ./spirv-bins/good.spv and input.txt from current directory]
./tritonspvc++.exe --spv <path to SPIRV>
