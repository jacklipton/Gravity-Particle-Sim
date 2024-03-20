cmake_minimum_required(VERSION 3.18)
project(MySDLApp LANGUAGES CXX CUDA)

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
set(CUDA_SOURCES
    kernels.cu
    main.cpp
)

set (SOURCES
    Particle.cpp
    Particle.hpp
    Particle_CUDA.cuh
)


#

# CUDA compile
add_executable(MySDLApp ${SOURCES} ${CUDA_SOURCES})

#Set CUDA_ARCHITECTURES property
set_property(TARGET MySDLApp PROPERTY CUDA_ARCHITECTURES OFF)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# Include SDL2 headers
target_include_directories(MySDLApp PRIVATE ${SDL2_INCLUDE_DIRS})

# Link SDL2 and SDL2_gfx libraries
target_link_libraries(MySDLApp PRIVATE ${SDL2_LIBRARIES} SDL2_gfx)

# Link CUDA libraries
target_link_libraries(MySDLApp PRIVATE ${CUDA_LIBRARIES})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S")
