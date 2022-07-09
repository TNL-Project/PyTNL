set( sources
      DistributedMesh.cpp
      DistributedMeshReaders.cpp
      DistributedMeshWriters.cpp
      tnl_mpi.cpp
)
pybind11_add_module(tnl_mpi ${sources})

# add TNL dependency
target_link_libraries(tnl_mpi PUBLIC stdc++fs)

# enable zlib and tinyxml2 (used by PVTUReader)
find_package(ZLIB)
find_package(tinyxml2)
if( ZLIB_FOUND )
   target_compile_definitions(tnl_mpi PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl_mpi PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl_mpi PUBLIC ${ZLIB_LIBRARIES})
endif()
if( tinyxml2_FOUND )
   target_compile_definitions(tnl_mpi PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(tnl_mpi PUBLIC tinyxml2::tinyxml2)
endif()

# enable MPI
target_compile_definitions(tnl_mpi PUBLIC "-DHAVE_MPI")
find_package(MPI COMPONENTS CXX REQUIRED)
target_link_libraries(tnl_mpi PUBLIC MPI::MPI_CXX)

install(TARGETS tnl_mpi DESTINATION ${PYTHON_SITE_PACKAGES_DIR})
