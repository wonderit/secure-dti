cmake_minimum_required(VERSION 3.5.1)  # CMake version check
project(secure_ecg)
# Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard
set(INCPATHS -I/Users/wonsuk/projects/local/include -Iinclude)
set(LDLIBS -lssl -lcrypto -lntl -lgmp -lm -ldl)
set(LDPATH -L/usr/local/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O1 ${INCPATHS} -march=native -std=c++11 -pthread -Wno-sign-compare")
message(STATUS "start running cmake...")

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES
        crypto.cpp
        connect.cpp
        aesstream.cpp
        mpc_int.cpp
        param.cpp
        ShareDataECG.cpp
        TrainSecureMNIST.cpp
        UnitTestClient.cpp
        TrainSecureDTI.cpp
        TrainSecureTTE.cpp
        GenerateKey.cpp
        AnalyzeFeature.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(UnitTestClient ${SOURCE_FILES})

include_directories(/usr/local/include/)
include_directories(/usr/local/include/eigen3)
include_directories(include)