set( sources
      containers.cpp
)
nanobind_add_module(_containers ${sources})

# enable position-independent code
set_target_properties(_containers PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# enable link-time optimization
set_target_properties(_containers PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION})

# add TNL and PyTNL
target_link_libraries(_containers PUBLIC PyTNL::PyTNL)

install(TARGETS _containers DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR}/pytnl)
