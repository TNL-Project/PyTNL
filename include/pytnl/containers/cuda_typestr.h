#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Backend.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/CudaManaged.h>
#include <TNL/TypeTraits.h>

// Maps a C++ scalar/complex type to its NumPy typestring code for the CUDA Array Interface.
// Format: endianness ('|' = N/A for 1-byte, '<' = little-endian) + kind + itemsize_in_bytes.
template< typename T >
constexpr std::string_view
cuda_typestr()
{
   using V = std::remove_const_t< T >;
   if constexpr( std::is_same_v< V, bool > ) {
      return "|b1";
   }
   else if constexpr( std::is_integral_v< V > ) {
      if constexpr( sizeof( V ) == 1 )
         return "|i1";
      else if constexpr( sizeof( V ) == 2 )
         return "<i2";
      else if constexpr( sizeof( V ) == 4 )
         return "<i4";
      else if constexpr( sizeof( V ) == 8 )
         return "<i8";
      else
         static_assert( sizeof( V ) == 0, "Unsupported integer size for __cuda_array_interface__" );
   }
   else if constexpr( std::is_floating_point_v< V > ) {
      if constexpr( sizeof( V ) == 4 )
         return "<f4";
      else if constexpr( sizeof( V ) == 8 )
         return "<f8";
      else
         static_assert( sizeof( V ) == 0, "Unsupported float size for __cuda_array_interface__" );
   }
   else if constexpr( TNL::is_complex_v< V > ) {
      if constexpr( sizeof( V ) == 8 )
         return "<c8";
      else if constexpr( sizeof( V ) == 16 )
         return "<c16";
      else
         static_assert( sizeof( V ) == 0, "Unsupported complex size for __cuda_array_interface__" );
   }
   else {
      static_assert( sizeof( V ) == 0, "Unsupported type for __cuda_array_interface__" );
   }
   // Unreachable — static_assert above guarantees a compile error for unsupported types.
   // The trailing return satisfies the compiler's requirement that all paths return a value.
   return "";
}
