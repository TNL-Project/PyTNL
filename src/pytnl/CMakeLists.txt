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
nanobind_add_module(meshes ${src_meshes})

# enable MPI
find_package(MPI COMPONENTS CXX REQUIRED)
target_compile_definitions(meshes PUBLIC "-DHAVE_MPI")
target_link_libraries(meshes PUBLIC MPI::MPI_CXX)

# set common properties
set(modules
    _containers
    matrices
    meshes
)
foreach(target IN ITEMS ${modules})
    # enable position-independent code
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

    # enable link-time optimization
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

    # add TNL and PyTNL
    target_link_libraries(${target} PUBLIC PyTNL::PyTNL)

    # install the module
    install(TARGETS ${target} DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
endforeach()
