set( sources
      matrices.cpp
)
nanobind_add_module(matrices ${sources})

# enable position-independent code
set_target_properties(matrices PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(matrices PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(matrices PUBLIC PyTNL::PyTNL)

install(TARGETS matrices DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
