set( sources
      Grid1D.cpp
      Grid2D.cpp
      Grid3D.cpp
      Mesh.cpp
      MeshReaders.cpp
      MeshWriters.cpp
      SparseMatrix.cpp
      String.cpp
      VTKTraits.cpp
      tnl.cpp
)
pybind11_add_module(tnl NO_EXTRAS ${sources})

# enable position-independent code
set_target_properties(tnl PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(tnl PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(tnl PUBLIC PyTNL::PyTNL)

install(TARGETS tnl DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
