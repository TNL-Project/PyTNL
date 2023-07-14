set( sources
      Grid1D.cpp
      Grid2D.cpp
      Grid3D.cpp
      Mesh.cpp
      MeshReaders.cpp
      MeshWriters.cpp
      Object.cpp
      SparseMatrix.cpp
      String.cpp
      VTKTraits.cpp
      tnl.cpp
)
pybind11_add_module(tnl ${sources})

# add TNL and PyTNL
target_link_libraries(tnl PUBLIC TNL::TNL_CXX PyTNL::PyTNL)

# enable zlib and tinyxml2 (used by VTUReader)
find_package(ZLIB)
find_package(tinyxml2)
if( ZLIB_FOUND )
   target_compile_definitions(tnl PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl PUBLIC ${ZLIB_LIBRARIES})
endif()
if( tinyxml2_FOUND )
   target_compile_definitions(tnl PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(tnl PUBLIC tinyxml2::tinyxml2)
endif()

install(TARGETS tnl DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
