cmake_minimum_required(VERSION 3.19)
project(depthInpainter)
set(CMAKE_TOOLCHAIN_FILE D:/lwj/data/vcpkg/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
endif (NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(depthInpainter main.cpp
        tgvOperator.cpp
        mat_vector.h
        SLIC.cpp
        SLIC.h
        GDBSCAN.cpp
        GDBSCAN.h
        RubustInpainter.cpp
        ImgSeg.cpp
        tgvAlgrothm.cpp colorize.cpp colorize.h SparseOptimizer.cpp SparseOptimizer.h utility.cpp utility.h outlinerRemove.cpp outlinerRemove.h)
target_link_libraries(depthInpainter ${OpenCV_LIBS}
        )
