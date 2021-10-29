cmake_minimum_required(VERSION 3.20)
project(depthInpainter)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
endif (NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(depthInpainter main.cpp
        tgvOperator.cpp
        mat_vector.h SLIC.cpp SLIC.h)
target_link_libraries(depthInpainter ${OpenCV_LIBS} )
