cmake_minimum_required(VERSION 3.26)
project(PlanetFormation2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -fopenmp -pthread")
add_executable(PlanetFormation2 main.cpp
        barnes_hut.h)

