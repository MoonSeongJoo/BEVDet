cmake_minimum_required(VERSION 3.28)
project(bevdet_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

file(GLOB_RECURSE REMOVE_CMAKE "build/*")
file(GLOB_RECURSE SOURCES
     "*.hpp"
     "*.cpp"
)
list(REMOVE_ITEM SOURCES ${REMOVE_CMAKE})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# Include the FetchContent module
include(FetchContent)
# Declare Eigen3 via FetchContent
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.3.9
)

FetchContent_MakeAvailable(eigen)

add_executable(bevdet_runtime ${SOURCES})
set_property(TARGET bevdet_runtime PROPERTY CXX_STANDARD 17)
target_link_libraries(bevdet_runtime zmq "${TORCH_LIBRARIES}")
target_include_directories(bevdet_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/x340_rt"
"${CMAKE_CURRENT_SOURCE_DIR}/Core"
"${CMAKE_CURRENT_SOURCE_DIR}/Service" 
"${CMAKE_CURRENT_SOURCE_DIR}/external"
"${CMAKE_CURRENT_SOURCE_DIR}/utils"
"${CMAKE_CURRENT_SOURCE_DIR}/utils"
${eigen_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR} )
target_compile_features(bevdet_runtime PRIVATE cxx_std_17)