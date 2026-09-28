# Builds the course PCL binding (pcl_python_merlab) as a Python extension module.
cmake_minimum_required(VERSION 3.16)
project(pcl_python_merlab LANGUAGES C CXX)   # C is needed: VTK (pulled in by PCL) looks for MPI_C

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters sample_consensus segmentation search kdtree
             features registration surface)
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(pcl_python_merlab NO_EXTRAS
  src/bindings.cpp src/features.cpp src/filters.cpp src/geometry.cpp src/registration.cpp src/surface_io.cpp)
target_include_directories(pcl_python_merlab PRIVATE src ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_python_merlab PRIVATE ${PCL_LIBRARIES})
