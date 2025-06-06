#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/tnl/Array.h>
#include <pytnl/tnl/Vector.h>

// external functions
void
export_Grid1D( nb::module_& m );
void
export_Grid2D( nb::module_& m );
void
export_Grid3D( nb::module_& m );
void
export_VTKTraits( nb::module_& m );
void
export_Meshes( nb::module_& m );
void
export_MeshReaders( nb::module_& m );
void
export_MeshWriters( nb::module_& m );
void
export_SparseMatrices( nb::module_& m );

template< typename T >
using _array = TNL::Containers::Array< T, TNL::Devices::Host, IndexType >;

template< typename T >
using _vector = TNL::Containers::Vector< T, TNL::Devices::Host, IndexType >;

// Python module definition
NB_MODULE( tnl, m )
{
   register_exceptions( m );

   // TODO: TNL::File

   export_Array< _array< double > >( m, "Array" );
   export_Vector< _array< double >, _vector< double > >( m, "Vector" );
   export_Array< _array< int > >( m, "Array_int" );
   export_Vector< _array< int >, _vector< int > >( m, "Vector_int" );
   export_Array< _array< bool > >( m, "Array_bool" );

   export_Grid1D( m );
   export_Grid2D( m );
   export_Grid3D( m );

   export_VTKTraits( m );

   export_Meshes( m );
   export_MeshReaders( m );
   export_MeshWriters( m );

   export_SparseMatrices( m );
}
