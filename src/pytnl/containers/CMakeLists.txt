set( sources
      containers.cpp
)
nanobind_add_module(containers ${sources})

# enable position-independent code
set_target_properties(containers PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(containers PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(containers PUBLIC PyTNL::PyTNL)

install(TARGETS containers DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
