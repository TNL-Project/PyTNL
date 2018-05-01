add_subdirectory( tnl )

set( headers
         exceptions.h
         RawIterator.h
         tnl_conversions.h
         tnl_indexing.h
         tnl_str_conversion.h
         tnl_tuple_conversion.h
         typedefs.h
)

install( FILES ${headers} DESTINATION "pytnl" )
