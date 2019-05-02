cmake_minimum_required(VERSION 3.10)

project(potential_backbone)
set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(       
	${PROJECT_SOURCE_DIR}/src
)

set(SOURCE_FILES src/main.cpp)   

find_package (Eigen3 3.3 REQUIRED)
add_executable (potential ${SOURCE_FILES})
target_link_libraries (potential Eigen3::Eigen)
