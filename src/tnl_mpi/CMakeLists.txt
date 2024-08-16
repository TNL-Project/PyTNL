set( sources
      DistributedMesh.cpp
      DistributedMeshReaders.cpp
      DistributedMeshWriters.cpp
      tnl_mpi.cpp
)
pybind11_add_module(tnl_mpi ${sources})

# enable link-time optimization
set_target_properties(tnl PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# add TNL and PyTNL
target_link_libraries(tnl_mpi PUBLIC PyTNL::PyTNL)

# enable MPI
find_package(MPI COMPONENTS CXX REQUIRED)
target_compile_definitions(tnl_mpi PUBLIC "-DHAVE_MPI")
target_link_libraries(tnl_mpi PUBLIC MPI::MPI_CXX)

install(TARGETS tnl_mpi DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
