set(src_containers containers/ArrayVector.cpp containers/StaticVector.cpp containers/NDArray.cpp containers/containers.cpp)
nanobind_add_module(_containers ${src_containers})
set(src_containers_cuda containers/ArrayVector.cu containers/NDArray.cu containers/containers.cu)
if(PyTNL_BUILD_CUDA)
    nanobind_add_module(_containers_cuda ${src_containers_cuda})
endif()
set(src_matrices matrices/matrices.cpp)
nanobind_add_module(_matrices ${src_matrices})
set(src_matrices_cuda matrices/matrices.cu)
if(PyTNL_BUILD_CUDA)
    nanobind_add_module(_matrices_cuda ${src_matrices_cuda})
endif()
set(src_meshes
    meshes/topologies.cpp
    meshes/VTKTraits.cpp
    meshes/Grid1D.cpp
    meshes/Grid2D.cpp
    meshes/Grid3D.cpp
    meshes/Mesh.cpp
    meshes/MeshReaders.cpp
    meshes/MeshWriters.cpp
    meshes/resolveMeshType.cpp
    meshes/DistributedMesh.cpp
    meshes/DistributedMeshReaders.cpp
    meshes/DistributedMeshWriters.cpp
    meshes/distributeSubentities.cpp
    meshes/meshes.cpp
)
set(src_meshes_cuda
    # VTKTraits has no GPU code
    meshes/Grid1D.cu
    meshes/Grid2D.cu
    meshes/Grid3D.cu
    meshes/Mesh.cu
    meshes/MeshReaders.cu
    meshes/MeshWriters.cu
    meshes/resolveMeshType.cu
    meshes/DistributedMesh.cu
    meshes/DistributedMeshReaders.cu
    meshes/DistributedMeshWriters.cu
    # distributeSubentities is host-only
    meshes/meshes.cu
)
nanobind_add_module(_meshes ${src_meshes})
set(src_solvers solvers/ode_methods.cpp solvers/solvers.cpp)
nanobind_add_module(_solvers ${src_solvers})
set(src_solvers_cuda solvers/solvers.cu)
if(PyTNL_BUILD_CUDA)
    nanobind_add_module(_solvers_cuda ${src_solvers_cuda})
endif()

# add dependencies
target_compile_definitions(_meshes PUBLIC "-DHAVE_ZLIB -DHAVE_TINYXML2")
target_link_libraries(_meshes PUBLIC ZLIB::ZLIB tinyxml2::tinyxml2)

if(PyTNL_BUILD_CUDA)
    nanobind_add_module(_meshes_cuda ${src_meshes_cuda})

    # add dependencies
    target_compile_definitions(_meshes_cuda PUBLIC "-DHAVE_ZLIB -DHAVE_TINYXML2")
    target_link_libraries(_meshes_cuda PUBLIC ZLIB::ZLIB tinyxml2::tinyxml2)
endif()

# define a list of modules
set(modules _containers _matrices _meshes _solvers)

# define a mapping for stub files
# (needed because the stubgen in recursive mode may generate multiple files
# when `.def_submodule` is used and cmake cannot automatically detect them)
set(module_stubs__containers _containers.pyi)
set(module_stubs__containers_cuda _containers_cuda.pyi)
set(module_stubs__matrices _matrices/__init__.pyi _matrices/formats.pyi)
set(module_stubs__matrices_cuda _matrices_cuda.pyi)
set(module_stubs__meshes _meshes/__init__.pyi _meshes/topologies.pyi)
set(module_stubs__meshes_cuda _meshes_cuda.pyi)
set(module_stubs__solvers _solvers/__init__.pyi _solvers/ode_methods.pyi)
set(module_stubs__solvers_cuda _solvers_cuda.pyi)

# define a mapping for dependencies between the modules
set(module_depends__containers)
set(module_depends__matrices _containers)
set(module_depends__meshes _containers)
set(module_depends__solvers _containers)
set(module_depends__solvers_cuda _containers_cuda _solvers _containers)

# add CUDA modules
if(PyTNL_BUILD_CUDA)
    list(APPEND modules _containers_cuda _matrices_cuda _meshes_cuda _solvers_cuda)
    set(module_depends__containers_cuda _containers)
    set(module_depends__matrices_cuda _containers_cuda _containers)
    set(module_depends__meshes_cuda _containers_cuda _meshes _containers)
endif()

# locate nanobind's stubgen.py (layout differs between system package and FetchContent)
if(EXISTS ${NB_DIR}/stubgen.py)
    set(NB_STUBGEN ${NB_DIR}/stubgen.py)
elseif(EXISTS ${NB_DIR}/src/stubgen.py)
    set(NB_STUBGEN ${NB_DIR}/src/stubgen.py)
else()
    message(FATAL_ERROR "Could not find stubgen.py under NB_DIR=${NB_DIR}")
endif()

# set common properties
foreach(target IN ITEMS ${modules})
    # enable position-independent code
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

    # enable link-time optimization
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

    # add TNL and PyTNL
    target_link_libraries(${target} PUBLIC PyTNL::PyTNL)

    # enable MPI
    target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
    target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

    # generate stub files for Python static type checking
    #
    # PYTHONPATH prioritizes the build directory over an existing editable
    # install, but that alone is not enough: scikit-build-core's editable
    # install registers a meta path finder via a .pth file in site-packages
    # that intercepts all "pytnl.*" imports and redirects them to the
    # installed .so files. After a nanobind version bump, those installed
    # .so files have a stale ABI, which breaks nanobind's cross-module type
    # lookup and causes stubgen to emit raw C++ RTTI names (e.g.
    # "TNL::Containers::Vector<...>") instead of Python-qualified names
    # (e.g. pytnl._containers.Vector_float) in the .pyi stubs.
    #
    # Passing -S disables site.py, so the .pth file is never processed and
    # the meta path finder is not installed. PYTHONPATH then resolves all
    # imports to the freshly built .so files. stubgen.py uses only stdlib
    # modules, so it runs fine without site-packages.
    #
    # We also set OpenMPI environment variables to suppress GPU accelerator
    # detection, which can fail at build time and cause MPI_Init to abort.
    set(_stubgen_args -q -r -m ${target})
    foreach(_py_path $<TARGET_FILE_DIR:${target}> $<TARGET_FILE_DIR:${target}>/..)
        list(APPEND _stubgen_args -i "${_py_path}")
    endforeach()
    # Use -O (output directory) instead of -o (output file) because
    # recursive mode may produce multiple .pyi files
    list(APPEND _stubgen_args -O "${CMAKE_CURRENT_BINARY_DIR}")

    add_custom_command(
        OUTPUT ${module_stubs_${target}}
        COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:${target}>/..:$<TARGET_FILE_DIR:${target}>
            OMPI_MCA_accelerator=null ${Python_EXECUTABLE} -S ${NB_STUBGEN} ${_stubgen_args}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS ${target} ${module_depends_${target}} "${NB_STUBGEN}"
    )
    add_custom_target(${target}_stub ALL DEPENDS ${module_stubs_${target}})

    # install the module
    install(TARGETS ${target} DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)

    # install the stub file(s)
    list(GET module_stubs_${target} 0 _first_stub)
    get_filename_component(_module_directory ${_first_stub} DIRECTORY)
    if(NOT _module_directory STREQUAL "")
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_module_directory} DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
    else()
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${module_stubs_${target}} DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
    endif()
endforeach()

# make the modules importable from the build directory
# (needed for the stub generation due to binary modules importing themselves as `pytnl._containers` etc.)
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

# one marker per module directory
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/py.typed)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
