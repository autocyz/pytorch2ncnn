cmake_minimum_required(VERSION 3.0)
project(2d23d)

set(CMAKE_CXX_STANDARD 11) 

set(NCNN_LIBS /home/cyz/data/Framework/ncnn/build/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS /home/cyz/data/Framework/ncnn/build/install/include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)  
endif()

include_directories(${NCNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp small.mem.h small.id.h)
target_link_libraries(main ${NCNN_LIBS} ${OpenCV_LIBS})
