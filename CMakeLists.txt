cmake_minimum_required(VERSION 3.10)

#Project name
project(JetClustering)

find_package(HDF5 REQUIRED COMPONENTS CXX)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(serialCL serialJetClustering.cpp)

#link
target_include_directories(serialCL PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(serialCL PRIVATE ${HDF5_LIBRARIES})