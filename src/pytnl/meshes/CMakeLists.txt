set( sources
      Grid1D.cpp
      Grid2D.cpp
      Grid3D.cpp
      Mesh.cpp
      MeshReaders.cpp
      MeshWriters.cpp
      VTKTraits.cpp
      meshes.cpp
)
nanobind_add_module(meshes ${sources})

# enable position-independent code
set_target_properties(meshes PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(meshes PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(meshes PUBLIC PyTNL::PyTNL)

install(TARGETS meshes DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
