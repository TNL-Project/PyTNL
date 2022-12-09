cmake_minimum_required(VERSION 3.18)
project(PyTNL
    LANGUAGES CXX
)

# Require C++14, and disable compiler-specific extensions (if possible).
foreach(lang CXX)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Werror=vla")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Enable link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Set the project's include path
include_directories(include)

# Set up TNL
add_subdirectory(libs/tnl)

# Set the install path for Python modules
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_SITE_PACKAGES_DIR lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# Set up pybind11
if( SYSTEM_PYBIND11 )
    find_package(pybind11 REQUIRED)
else()
    add_subdirectory(libs/pybind11)
endif()

# Add subdirectories
add_subdirectory(include)
add_subdirectory(src)
