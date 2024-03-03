cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
project(MySDLProject)

# Find SDL2
find_package(SDL2 REQUIRED)

# Include directory for SDL2_gfx
include_directories(${CMAKE_SOURCE_DIR}/path/to/SDL2_gfx)

# Link directory for SDL2_gfx (if necessary)
link_directories(${CMAKE_SOURCE_DIR}/path/to/SDL2_gfx/lib)

# Add executable
add_executable(MySDLApp main.cpp
                Particle.cpp
                Particle.hpp)

# Include SDL2 headers
target_include_directories(MySDLApp PRIVATE ${SDL2_INCLUDE_DIRS})

# Link SDL2 and SDL2_gfx libraries
target_link_libraries(MySDLApp PRIVATE ${SDL2_LIBRARIES} SDL2_gfx)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S")
