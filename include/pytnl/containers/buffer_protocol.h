#pragma once

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include <pytnl/pytnl.h>

namespace pytnl::containers::buffer_protocol
{

struct BufferInfo
{
   std::vector< Py_ssize_t > shape;
   std::vector< Py_ssize_t > strides;

   explicit BufferInfo( int ndim )
   : shape( static_cast< std::size_t >( ndim ) ),
     strides( static_cast< std::size_t >( ndim ) )
   {}
};

template< typename T >
constexpr const char*
pybuffer_format()
{
   using U = std::remove_cv_t< T >;
   if constexpr( std::is_same_v< U, bool > )
      return "?";
   else if constexpr( std::is_same_v< U, std::int8_t > )
      return "b";
   else if constexpr( std::is_same_v< U, std::uint8_t > )
      return "B";
   else if constexpr( std::is_same_v< U, std::int16_t > )
      return "h";
   else if constexpr( std::is_same_v< U, std::uint16_t > )
      return "H";
   else if constexpr( std::is_same_v< U, std::int32_t > )
      return "i";
   else if constexpr( std::is_same_v< U, std::uint32_t > )
      return "I";
   else if constexpr( std::is_same_v< U, std::int64_t > )
      return "l";
   else if constexpr( std::is_same_v< U, std::uint64_t > )
      return "L";
   else if constexpr( std::is_same_v< U, float > )
      return "f";
   else if constexpr( std::is_same_v< U, double > )
      return "d";
   else
      return nullptr;
}

inline bool
checked_cast_to_py_ssize( std::size_t value, Py_ssize_t& out )
{
   constexpr std::size_t max_py =
      static_cast< std::size_t >( std::numeric_limits< Py_ssize_t >::max() );
   if( value > max_py )
      return false;
   out = static_cast< Py_ssize_t >( value );
   return true;
}

inline bool
checked_mul_py_ssize( Py_ssize_t a, Py_ssize_t b, Py_ssize_t& out )
{
   if( a < 0 || b < 0 )
      return false;
   if( a == 0 || b == 0 ) {
      out = 0;
      return true;
   }
   if( a > std::numeric_limits< Py_ssize_t >::max() / b )
      return false;
   out = a * b;
   return true;
}

template< typename DeviceType >
constexpr bool
is_host_device_v = std::is_same_v< DeviceType, TNL::Devices::Host >;

template< typename ArrayType >
int
array_getbuffer( PyObject* exporter, Py_buffer* view, int flags )
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;

   if( view == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Py_buffer view cannot be NULL" );
      return -1;
   }

   if constexpr( ! is_host_device_v< DeviceType > ) {
      PyErr_SetString( PyExc_BufferError,
                       "Buffer protocol is only available for host device arrays. "
                       "Use __dlpack__ / __cuda_array_interface__ for non-host memory." );
      return -1;
   }

   constexpr const char* fmt = pybuffer_format< ValueType >();
   if( ( flags & PyBUF_FORMAT ) && fmt == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Unsupported ValueType for Python buffer format string" );
      return -1;
   }

   ArrayType* obj = nb::inst_ptr< ArrayType >( nb::handle( exporter ) );
   BufferInfo* info = new ( std::nothrow ) BufferInfo( 1 );
   if( info == nullptr ) {
      PyErr_NoMemory();
      return -1;
   }

   Py_ssize_t size_py = 0;
   if( ! checked_cast_to_py_ssize( static_cast< std::size_t >( obj->getSize() ), size_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Array size does not fit into Py_ssize_t" );
      return -1;
   }

   Py_ssize_t itemsize_py = 0;
   if( ! checked_cast_to_py_ssize( sizeof( std::remove_cv_t< ValueType > ), itemsize_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Item size does not fit into Py_ssize_t" );
      return -1;
   }

   Py_ssize_t len_py = 0;
   if( ! checked_mul_py_ssize( size_py, itemsize_py, len_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Total buffer byte size overflow" );
      return -1;
   }

   info->shape[ 0 ] = size_py;
   info->strides[ 0 ] = itemsize_py;

   view->buf = const_cast< void* >( static_cast< const void* >( obj->getData() ) );
   view->obj = exporter;
   view->len = len_py;
   view->readonly = std::is_const_v< ValueType > ? 1 : 0;
   view->itemsize = itemsize_py;
   view->format = const_cast< char* >( fmt );
   view->ndim = 1;
   view->shape = info->shape.data();
   view->strides = info->strides.data();
   view->suboffsets = nullptr;
   view->internal = info;

   Py_INCREF( exporter );
   return 0;
}

template< typename NDArrayType >
int
ndarray_getbuffer( PyObject* exporter, Py_buffer* view, int flags )
{
   using ValueType = typename NDArrayType::ValueType;
   using DeviceType = typename NDArrayType::DeviceType;

   if( view == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Py_buffer view cannot be NULL" );
      return -1;
   }

   if constexpr( ! is_host_device_v< DeviceType > ) {
      PyErr_SetString( PyExc_BufferError,
                       "Buffer protocol is only available for host device arrays. "
                       "Use __dlpack__ / __cuda_array_interface__ for non-host memory." );
      return -1;
   }

   constexpr const char* fmt = pybuffer_format< ValueType >();
   if( ( flags & PyBUF_FORMAT ) && fmt == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Unsupported ValueType for Python buffer format string" );
      return -1;
   }

   NDArrayType* obj = nb::inst_ptr< NDArrayType >( nb::handle( exporter ) );

   constexpr int ndim = static_cast< int >( NDArrayType::getDimension() );
   static_assert( ndim > 0, "NDArray dimension must be positive" );

   BufferInfo* info = new ( std::nothrow ) BufferInfo( ndim );
   if( info == nullptr ) {
      PyErr_NoMemory();
      return -1;
   }

   Py_ssize_t itemsize_py = 0;
   if( ! checked_cast_to_py_ssize( sizeof( std::remove_cv_t< ValueType > ), itemsize_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Item size does not fit into Py_ssize_t" );
      return -1;
   }

   const auto sizes = obj->getSizes();

   Py_ssize_t total_elems = 1;
   for( int i = 0; i < ndim; i++ ) {
      Py_ssize_t dim_py = 0;
      if( ! checked_cast_to_py_ssize( static_cast< std::size_t >( sizes[ static_cast< std::size_t >( i ) ] ), dim_py ) ) {
         delete info;
         PyErr_SetString( PyExc_OverflowError, "NDArray dimension size does not fit into Py_ssize_t" );
         return -1;
      }
      info->shape[ static_cast< std::size_t >( i ) ] = dim_py;

      if( ! checked_mul_py_ssize( total_elems, dim_py, total_elems ) ) {
         delete info;
         PyErr_SetString( PyExc_OverflowError, "Total NDArray element count overflow" );
         return -1;
      }
   }

   // C-contiguous strides
   info->strides[ static_cast< std::size_t >( ndim - 1 ) ] = itemsize_py;
   for( int i = ndim - 2; i >= 0; i-- ) {
      Py_ssize_t stride = 0;
      if( ! checked_mul_py_ssize( info->strides[ static_cast< std::size_t >( i + 1 ) ],
                                  info->shape[ static_cast< std::size_t >( i + 1 ) ],
                                  stride ) ) {
         delete info;
         PyErr_SetString( PyExc_OverflowError, "Stride computation overflow" );
         return -1;
      }
      info->strides[ static_cast< std::size_t >( i ) ] = stride;
   }

   Py_ssize_t len_py = 0;
   if( ! checked_mul_py_ssize( total_elems, itemsize_py, len_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Total buffer byte size overflow" );
      return -1;
   }

   view->buf = const_cast< void* >( static_cast< const void* >( obj->getData() ) );
   view->obj = exporter;
   view->len = len_py;
   view->readonly = std::is_const_v< ValueType > ? 1 : 0;
   view->itemsize = itemsize_py;
   view->format = const_cast< char* >( fmt );
   view->ndim = ndim;
   view->shape = info->shape.data();
   view->strides = info->strides.data();
   view->suboffsets = nullptr;
   view->internal = info;

   Py_INCREF( exporter );
   return 0;
}

template< typename StaticVectorType >
int
static_vector_getbuffer( PyObject* exporter, Py_buffer* view, int flags )
{
   using ValueType = typename StaticVectorType::ValueType;

   if( view == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Py_buffer view cannot be NULL" );
      return -1;
   }

   constexpr const char* fmt = pybuffer_format< ValueType >();
   if( ( flags & PyBUF_FORMAT ) && fmt == nullptr ) {
      PyErr_SetString( PyExc_BufferError, "Unsupported ValueType for Python buffer format string" );
      return -1;
   }

   StaticVectorType* obj = nb::inst_ptr< StaticVectorType >( nb::handle( exporter ) );
   BufferInfo* info = new ( std::nothrow ) BufferInfo( 1 );
   if( info == nullptr ) {
      PyErr_NoMemory();
      return -1;
   }

   Py_ssize_t size_py = 0;
   if( ! checked_cast_to_py_ssize( static_cast< std::size_t >( StaticVectorType::getSize() ), size_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "StaticVector size does not fit into Py_ssize_t" );
      return -1;
   }

   Py_ssize_t itemsize_py = 0;
   if( ! checked_cast_to_py_ssize( sizeof( std::remove_cv_t< ValueType > ), itemsize_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Item size does not fit into Py_ssize_t" );
      return -1;
   }

   Py_ssize_t len_py = 0;
   if( ! checked_mul_py_ssize( size_py, itemsize_py, len_py ) ) {
      delete info;
      PyErr_SetString( PyExc_OverflowError, "Total buffer byte size overflow" );
      return -1;
   }

   info->shape[ 0 ] = size_py;
   info->strides[ 0 ] = itemsize_py;

   view->buf = const_cast< void* >( static_cast< const void* >( obj->getData() ) );
   view->obj = exporter;
   view->len = len_py;
   view->readonly = std::is_const_v< ValueType > ? 1 : 0;
   view->itemsize = itemsize_py;
   view->format = const_cast< char* >( fmt );
   view->ndim = 1;
   view->shape = info->shape.data();
   view->strides = info->strides.data();
   view->suboffsets = nullptr;
   view->internal = info;

   Py_INCREF( exporter );
   return 0;
}

inline void
releasebuffer( PyObject* /* exporter */, Py_buffer* view )
{
   delete static_cast< BufferInfo* >( view->internal );
}

template< typename ArrayType >
inline PyType_Slot*
array_buffer_slots()
{
   static PyType_Slot slots[] = {
      { Py_bf_getbuffer, reinterpret_cast< void* >( &array_getbuffer< ArrayType > ) },
      { Py_bf_releasebuffer, reinterpret_cast< void* >( &releasebuffer ) },
      { 0, nullptr }
   };
   return slots;
}

template< typename NDArrayType >
inline PyType_Slot*
ndarray_buffer_slots()
{
   static PyType_Slot slots[] = {
      { Py_bf_getbuffer, reinterpret_cast< void* >( &ndarray_getbuffer< NDArrayType > ) },
      { Py_bf_releasebuffer, reinterpret_cast< void* >( &releasebuffer ) },
      { 0, nullptr }
   };
   return slots;
}

template< typename StaticVectorType >
inline PyType_Slot*
static_vector_buffer_slots()
{
   static PyType_Slot slots[] = {
      { Py_bf_getbuffer, reinterpret_cast< void* >( &static_vector_getbuffer< StaticVectorType > ) },
      { Py_bf_releasebuffer, reinterpret_cast< void* >( &releasebuffer ) },
      { 0, nullptr }
   };
   return slots;
}

}  // namespace pytnl::containers::buffer_protocol