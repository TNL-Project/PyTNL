add_subdirectory( tnl )

install( DIRECTORY . DESTINATION "include/pytnl"
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )
