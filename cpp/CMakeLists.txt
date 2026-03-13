cmake_minimum_required(VERSION 3.18)
project(microalpha_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use modern Python discovery to avoid pybind11 picking the wrong interpreter
set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cpp
    src/features.cpp
    src/bindings.cpp
)

target_include_directories(_cpp PRIVATE include)

if(MSVC)
    target_compile_options(_cpp PRIVATE /O2 /W4)
else()
    target_compile_options(_cpp PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()