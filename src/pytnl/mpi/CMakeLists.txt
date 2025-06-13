set( sources
      DistributedMesh.cpp
      DistributedMeshReaders.cpp
      DistributedMeshWriters.cpp
      mpi.cpp
)
nanobind_add_module(mpi ${sources})

# enable position-independent code
set_target_properties(mpi PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(mpi PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(mpi PUBLIC PyTNL::PyTNL)

# enable MPI
find_package(MPI COMPONENTS CXX REQUIRED)
target_compile_definitions(mpi PUBLIC "-DHAVE_MPI")
target_link_libraries(mpi PUBLIC MPI::MPI_CXX)

install(TARGETS mpi DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
