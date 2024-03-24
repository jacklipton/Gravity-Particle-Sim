cmake_minimum_required(VERSION 3.12)
project(Fireworks_with_cuda)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture (change according to your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
set(CMAKE_CC_COMPILER "/usr/bin/gcc-9")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir/usr/bin/g++-9")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir/usr/bin/gcc-9")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Add CUDA source files
set(CUDA_SOURCE_FILES
    main.cu
    kernels.cu
    Particle_CUDA.cuh
)

# Add C++ source files
set(CPP_SOURCE_FILES
    # Add your C++ source files here
    Particle.cpp
    Particle.hpp
)

# Add SDL2 and SDLgfx libraries
find_package(SDL2 REQUIRED)
find_package(SDL2_gfx REQUIRED)

# Add include directories
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${SDL2_GFX_INCLUDE_DIRS})

# Add executable
cuda_add_executable(Fireworks_with_cuda ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})

# Link libraries
target_link_libraries(Fireworks_with_cuda ${SDL2_LIBRARIES} ${SDL2_GFX_LIBRARIES})