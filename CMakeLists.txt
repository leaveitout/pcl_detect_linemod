cmake_minimum_required(VERSION 3.0)
project(pcl_detect_linemod)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

find_package(PCL 1.7 REQUIRED)
#find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_HOME_DIRECTORY}/include)

set(SOURCE_FILES src/main.cpp)

add_executable(pcl_detect_linemod ${SOURCE_FILES})

target_link_libraries(pcl_detect_linemod ${PCL_LIBRARIES})
#target_link_libraries(pcl_detect_linemod ${OpenCV_LIBS})

