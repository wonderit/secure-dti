cmake_minimum_required(VERSION 3.13)  # CMake version check
project(secure_ecg)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 11)            # Enable c++14 standard

set(INCPATHS -I~/projects/local/include)
set(LDLIBS -lssl -lcrypto -lntl -lgmp -lm -ldl)
set(LDPATH -L~/projects/local/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 ${INCPATHS} -march=native -std=c++11 -pthread -Wno-sign-compare")
message(STATUS "start running cmake...")

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES
        crypto.cpp
        connect.cpp
        aesstream.cpp
        mpc.cpp
        param.cpp
        UnitTestClient.cpp
#        ShareData.cpp
#        TrainSecureMNIST.cpp
#        TrainSecureDTI.cpp
        TrainSecureECG.cpp
#        GenerateKey.cpp
#        AnalyzeFeature.cpp
#        boost_test.cpp
        )

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(UnitTestClient ${SOURCE_FILES})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(/usr/local/include /usr/local/lib /usr/local/opt/openssl/lib)
find_package(OpenSSL REQUIRED)
if( OpenSSL_FOUND )
    include_directories(${OPENSSL_INCLUDE_DIRS})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "Using OpenSSL ${OPENSSL_VERSION} ${OPENSSL_LIBRARIES}")
endif()

find_library(NTL_LIB ntl)
if(NOT NTL_LIB)
    message(FATAL_ERROR "ntl library not found.  Rerun cmake with -DCMAKE_PREFIX_PATH=\"<path to lib1>;<path to lib2>\"")
endif()

find_library(GMP_LIB gmp)
if(NOT GMP_LIB)
    message(FATAL_ERROR "gmp library not found.  Rerun cmake with -DCMAKE_PREFIX_PATH=\"<path to lib1>;<path to lib2>\"")
endif()
message(${NTL_LIB}, ${GMP_LIB})

target_link_libraries(UnitTestClient ${OPENSSL_LIBRARIES} ${NTL_LIB} ${M_LIB} ${GMP_LIB})