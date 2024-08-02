cmake_minimum_required(VERSION 3.28)
project(PyTNL
    LANGUAGES CXX
)

# check if PyTNL is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
   set(PyTNL_TOP_LEVEL_PROJECT ON)
   set(PyTNL_SYSTEM "")
else()
   set(PyTNL_TOP_LEVEL_PROJECT OFF)
   set(PyTNL_SYSTEM SYSTEM)
endif()

# make cache variables for install destinations
include(GNUInstallDirs)

# install paths relative to the cmake's prefix
set(PyTNL_TARGET_CMAKE_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Require C++17, and disable compiler-specific extensions (if possible).
foreach(lang CXX)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# make cache variable so it can be used in downstream projects
set(PyTNL_INCLUDE_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/include"
    CACHE INTERNAL "Directories where PyTNL headers are located")

# create the exported targets
add_library(PyTNL INTERFACE)
# aliases to match exported targets
add_library(PyTNL::PyTNL ALIAS PyTNL)

# add the include directory to the interface
target_include_directories(PyTNL ${PyTNL_SYSTEM}
                           INTERFACE $<BUILD_INTERFACE:${PyTNL_INCLUDE_DIRS}>
                                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# add compiler features to the interface
target_compile_features(PyTNL INTERFACE cxx_std_17)

if(PyTNL_TOP_LEVEL_PROJECT)
    # export the interface targets
    install(TARGETS PyTNL EXPORT PyTNLTargets)
    # install a CMake file for the interface target
    install(EXPORT PyTNLTargets
            NAMESPACE PyTNL::
            DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
    # install the PyTNLConfig.cmake file
    include(CMakePackageConfigHelpers)
    configure_package_config_file("cmake/${PROJECT_NAME}Config.cmake.in"
                                  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                  INSTALL_DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
endif()

# Add dependencies
include(FetchContent)
FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    GIT_TAG main
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(TNL)

# Set the install path for Python modules
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
set(PyTNL_PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master
    SYSTEM
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(pybind11)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(src)
