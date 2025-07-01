set(src_containers
    containers/containers.cpp
)
nanobind_add_module(_containers ${src_containers})
set(src_matrices
    matrices/matrices.cpp
)
nanobind_add_module(matrices ${src_matrices})
set(src_meshes
    meshes/Grid1D.cpp
    meshes/Grid2D.cpp
    meshes/Grid3D.cpp
    meshes/Mesh.cpp
    meshes/MeshReaders.cpp
    meshes/MeshWriters.cpp
    meshes/VTKTraits.cpp
    meshes/resolveMeshType.cpp
    meshes/DistributedMesh.cpp
    meshes/DistributedMeshReaders.cpp
    meshes/DistributedMeshWriters.cpp
    meshes/distributeSubentities.cpp
    meshes/meshes.cpp
)
nanobind_add_module(_meshes ${src_meshes})

# add dependencies
target_compile_definitions(_meshes PUBLIC "-DHAVE_ZLIB -DHAVE_TINYXML2")
target_link_libraries(_meshes PUBLIC ZLIB::ZLIB tinyxml2::tinyxml2)

# enable MPI
find_package(MPI COMPONENTS CXX REQUIRED)
target_compile_definitions(_meshes PUBLIC "-DHAVE_MPI")
target_link_libraries(_meshes PUBLIC MPI::MPI_CXX)

# define a list of modules
set(modules
    _containers
    matrices
    _meshes
)

# define a mapping for dependencies between the modules
set(module_depends__containers)
set(module_depends_matrices _containers)
set(module_depends__meshes _containers)

# set common properties
foreach(target IN ITEMS ${modules})
    # enable position-independent code
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

    # enable link-time optimization
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

    # add TNL and PyTNL
    target_link_libraries(${target} PUBLIC PyTNL::PyTNL)

    # generate stub files for Python static type checking
    nanobind_add_stub(
        ${target}_stub
        MODULE ${target}
        OUTPUT ${target}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${target}> $<TARGET_FILE_DIR:${target}>/..
        DEPENDS ${target} ${module_depends_${target}}
    )

    # install the module and stub file
    install(TARGETS ${target} DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${target}.pyi DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
endforeach()

# make the modules importable from the build directory
# (needed for the stub generation due to binary modules importing themselves as `pytnl._containers` etc.)
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

# one marker per module directory
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/py.typed)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
