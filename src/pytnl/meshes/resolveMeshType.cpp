#include <pytnl/pytnl.h>

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

void
export_resolveMeshType( nb::module_& m )
{
   using MeshReader = TNL::Meshes::Readers::MeshReader;

   auto resolveMeshType = []( const std::string& file_name, const std::string& file_format = "auto" )  //
      -> nb::typed< nb::tuple, MeshReader, nb::type_object >
   {
      // NOTE: We cannot get the reader with TNL::Meshes::resolveMeshType,
      // because it exposes it only *by value* and we can't make a copy of
      // that (because it is an abstract base class MeshReader). Thus we
      // need to reimplement and use TNL::Meshes::Readers::getMeshReader
      // which returns a std::shared_ptr which is ok.
      std::shared_ptr< TNL::Meshes::Readers::MeshReader > reader =
         TNL::Meshes::Readers::getMeshReader( file_name, file_format );
      if( reader == nullptr )
         return nb::make_tuple( nb::none(), nb::none() );

      reader->detectMesh();
      reader->forceRealType( TNL::getType< RealType >() );
      // FIXME: hardcoded type name
      reader->forceGlobalIndexType( "std::int64_t" );

      nb::object py_mesh = nb::none();
      auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> bool
      {
         py_mesh = nb::cast( mesh );
         return true;
      };

      bool result = false;
      if( reader->getMeshType() == "Meshes::Grid" || reader->getMeshType() == "Meshes::DistributedGrid" )
         result = TNL::Meshes::GridTypeResolver< PyTNLConfigTag, DeviceType >::run( *reader, wrapper );
      else if( reader->getMeshType() == "Meshes::Mesh" || reader->getMeshType() == "Meshes::DistributedMesh" )
         result = TNL::Meshes::MeshTypeResolver< PyTNLConfigTag, DeviceType >::run( *reader, wrapper );
      else {
         throw std::runtime_error( "The mesh type " + reader->getMeshType() + " is not supported." );
      }
      if( ! result )
         throw std::runtime_error( "Failed to resolve mesh type from given file." );

      nb::object py_reader = nb::cast( reader );
      return nb::make_tuple( std::move( py_reader ), std::move( py_mesh ) );
   };

   auto resolveAndLoadMesh = [ resolveMeshType ]( const std::string& file_name,
                                                  const std::string& file_format = "auto" )  //
      -> nb::typed< nb::tuple, MeshReader, nb::type_object >
   {
      nb::tuple reader_and_mesh = resolveMeshType( file_name, file_format );
      reader_and_mesh[ 0 ].attr( "loadMesh" )( reader_and_mesh[ 1 ] );
      return reader_and_mesh;
   };

   m.def( "resolveMeshType",
          resolveMeshType,
          nb::arg( "file_name" ),
          nb::kw_only(),
          nb::arg( "file_format" ) = "auto",
          "Returns a `(reader, mesh)` pair where `reader` is initialized "
          "with the given file name (using `getMeshReader`) and `mesh` is empty." );

   m.def( "resolveAndLoadMesh",
          resolveAndLoadMesh,
          nb::arg( "file_name" ),
          nb::kw_only(),
          nb::arg( "file_format" ) = "auto",
          "Returns a `(reader, mesh)` pair where `reader` is initialized "
          "with the given file name (using `getMeshReader`) and `mesh` contains "
          "the mesh loaded from the given file (using `reader.loadMesh(mesh)`)." );
}
