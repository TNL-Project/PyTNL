set( sources
      Grid1D.cpp
      Grid2D.cpp
      Grid3D.cpp
#      Mesh.cpp
      Object.cpp
      SparseMatrix.cpp
      String.cpp
      tnl.cpp
)
pybind11_add_module( pytnl ${sources} )

# link against tnl.so
target_link_libraries( pytnl PRIVATE tnl )

# rename the shared library to tnl.cpython-XXm-x86_64-linux-gnu.so
set_target_properties( pytnl PROPERTIES LIBRARY_OUTPUT_NAME tnl )

# We have bindings for unsafe objects (e.g. Array::operator[]) where assertion
# is the only safeguard, so we need to translate the TNL::AssertionError to
# Python's AssertionError.
# NDEBUG is defined in the global CMAKE_CXX_FLAGS and cannot be easily removed
# per-target, so we need to undefine it by passing -U NDEBUG.
target_compile_options( pytnl PRIVATE -U NDEBUG -D TNL_THROW_ASSERTION_ERROR )

install( TARGETS pytnl DESTINATION ${PYTHON_SITE_PACKAGES_DIR} )
