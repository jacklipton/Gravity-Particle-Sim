cmake_minimum_required(VERSION 3.18)
project(MySDLApp LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SDL2
find_package(SDL2 REQUIRED)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directory for SDL2_gfx
include_directories(${CMAKE_SOURCE_DIR}/path/to/SDL2_gfx)

# Link directory for SDL2_gfx (if necessary)
link_directories(${CMAKE_SOURCE_DIR}/path/to/SDL2_gfx/lib)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Set the C compiler
set(CMAKE_C_COMPILER /usr/bin/gcc-9)

# Set the C++ compiler
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

# CUDA flags
set(CMAKE_CUDA_FLAGS ${CUDA_CUDA_FLAGS} -std=c++14)

# Add CUDA sources
set (CUDA_SOURCES
    kernels.cu
    Particle_CUDA.cuh
    main.cu
)

# Add C++ sources
set (SOURCES

    Particle.cpp
    Particle.hpp
)

# Add CUDA kernel file to the project
cuda_add_library(ParticlesKernels ${CUDA_SOURCES})

# CUDA compile
add_executable(MySDLApp ${SOURCES})
target_link_libraries(MySDLApp PRIVATE ParticlesKernels)

# Set CUDA_ARCHITECTURES property
set_property(TARGET ParticlesKernels PROPERTY CUDA_ARCHITECTURES OFF)
set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CUDA)

# Include SDL2 headers
target_include_directories(MySDLApp PRIVATE ${SDL2_INCLUDE_DIRS})

# Link SDL2 and SDL2_gfx libraries
target_link_libraries(MySDLApp PRIVATE ${SDL2_LIBRARIES} SDL2_gfx)

# Link CUDA libraries
target_link_libraries(MySDLApp PRIVATE ${CUDA_LIBRARIES})
