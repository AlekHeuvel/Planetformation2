cmake_minimum_required(VERSION 3.26)
project(PlanetFormation)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
add_executable(PlanetFormation main.cpp
        barnes_hut.h)
