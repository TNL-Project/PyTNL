#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/make_iterator.h>

namespace nb = nanobind;

/* Reimplementation of nb::init but without list-initialization of the class,
 * which is not suitable for us due to using constructors with the
 * std::initializer_list parameter.
 *
 * Bug report: https://github.com/wjakob/nanobind/issues/1074
 */
template< typename... Args >
struct my_init : nb::def_visitor< my_init< Args... > >
{
   template< typename T, typename... Ts >
   friend class nb::class_;

   NB_INLINE
   my_init() {}

private:
   template< typename Class, typename... Extra >
   NB_INLINE static void
   execute( Class& cl, const Extra&... extra )
   {
      using Type = typename Class::Type;
      using Alias = typename Class::Alias;
      cl.def(
         "__init__",
         []( nb::pointer_and_handle< Type > v, Args... args )
         {
            if constexpr( ! std::is_same_v< Type, Alias > && std::is_constructible_v< Type, Args... > ) {
               if( ! nb::detail::nb_inst_python_derived( v.h.ptr() ) ) {
                  new( v.p ) Type( (nb::detail::forward_t< Args >) args... );
                  return;
               }
            }
            new( (void*) v.p ) Alias( (nb::detail::forward_t< Args >) args... );
         },
         extra... );
   }
};
