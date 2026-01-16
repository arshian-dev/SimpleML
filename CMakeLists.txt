cmake_minimum_required(VERSION 3.10)
project(SimpleML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create library
add_library(simple_ml ${SOURCES})

# Create executable for testing/examples
add_executable(ml_example examples/main.cpp)
target_link_libraries(ml_example PRIVATE simple_ml)

