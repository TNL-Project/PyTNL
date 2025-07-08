cmake_minimum_required(VERSION 3.28)
project(PyTNL
    LANGUAGES CXX
)

# Declare all CMake options for the project
option(PyTNL_EXPORT_INTERFACE_TARGETS "Instruct CMake to generate rules to export the interface target" ${PROJECT_IS_TOP_LEVEL})
option(PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization (IPO/LTO) for PyTNL targets" ON)

# make cache variables for install destinations
include(GNUInstallDirs)

# install paths relative to the cmake's prefix
if (SKBUILD_DATA_DIR)
    set(PyTNL_TARGET_CMAKE_DIRECTORY "${SKBUILD_DATA_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(PyTNL_TARGET_CMAKE_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif()

# Set the install path for Python modules
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
if (SKBUILD_PLATLIB_DIR)
    set(PyTNL_PYTHON_SITE_PACKAGES_DIR "${SKBUILD_PLATLIB_DIR}")
else()
    # Make cache variable so it can be used in downstream projects
    set(PyTNL_PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
        CACHE INTERNAL "Path where PyTNL Python packages are installed")
endif()

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
if(PROJECT_IS_TOP_LEVEL)
    set(PyTNL_SYSTEM "")
else()
    set(PyTNL_SYSTEM SYSTEM)
endif()
target_include_directories(PyTNL ${PyTNL_SYSTEM}
                           INTERFACE $<BUILD_INTERFACE:${PyTNL_INCLUDE_DIRS}>
                                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# add compiler features to the interface
target_compile_features(PyTNL INTERFACE cxx_std_17)

if(PyTNL_EXPORT_INTERFACE_TARGETS)
    # export the interface targets
    install(TARGETS PyTNL EXPORT PyTNLTargets)
    # install a CMake file for the interface target
    install(EXPORT PyTNLTargets
            NAMESPACE PyTNL::
            DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
    # install the PyTNLConfig.cmake file
    include(CMakePackageConfigHelpers)
    configure_package_config_file("${PROJECT_NAME}Config.cmake.in"
                                  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                  INSTALL_DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            DESTINATION ${PyTNL_TARGET_CMAKE_DIRECTORY})
endif()

# Set position independent code for all targets, including dependencies
# downloaded and built by FetchContent. Otherwise, linking to static libraries
# without PIC fails.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find or fetch dependencies
include(FetchContent)
FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    # TNL does not have any releases
    GIT_TAG d08c4deccd86d9b0075e1cf5f6d91b0176dc80df
    EXCLUDE_FROM_ALL
)
set(TNL_EXPORT_INTERFACE_TARGETS ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(TNL)

FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.7.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 11.0.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(tinyxml2_BUILD_TESTING OFF)
FetchContent_MakeAvailable(tinyxml2)

find_package(ZLIB REQUIRED)

# Add dependencies to the interface target
target_link_libraries(PyTNL INTERFACE TNL::TNL)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(src)
