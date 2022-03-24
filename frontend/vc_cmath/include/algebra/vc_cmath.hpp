/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/cmath.hpp"
#include "algebra/math/vc.hpp"
#include "algebra/storage/vc.hpp"

/// @name Operators on @c algebra::vc types
/// @{

using algebra::cmath::operator*;
using algebra::cmath::operator-;
using algebra::cmath::operator+;

/// @}

namespace algebra {
namespace vc {

/// @name cmath based transforms on @c algebra::vc types
/// @{

// Pull in the definitions needed by the cmath transforms, into this namespace.
using math::cross;
using math::perp;
using math::phi;

template <typename T>
using transform3 =
    cmath::transform3<std::size_t, vc::storage_type, T,
                      Vc::array<Vc::array<T, 4>, 4>,
                      cmath::element_getter<std::size_t, Vc::array, T>,
                      cmath::block_getter<std::size_t, Vc::array, T>,
                      vc::vector3<T>, vc::point2<T>>;
template <typename T>
using cartesian2 = cmath::cartesian2<transform3<T>>;
template <typename T>
using polar2 = cmath::polar2<transform3<T>>;
template <typename T>
using cylindrical2 = cmath::cylindrical2<transform3<T>>;

/// @}

}  // namespace vc

namespace getter {

/// @name Getter functions on @c algebra::vc types
/// @{

using cmath::eta;
using cmath::norm;
using cmath::perp;
using cmath::phi;
using cmath::theta;

using vc::math::eta;
using vc::math::norm;
using vc::math::perp;
using vc::math::phi;
using vc::math::theta;

/// @|

/// Function extracting a slice from the matrix used by
/// @c algebra::vc::transform3
template <std::size_t SIZE, std::size_t ROWS, std::size_t COLS,
          typename scalar_t>
ALGEBRA_HOST_DEVICE inline vc::storage_type<scalar_t, SIZE> vector(
    const vc::matrix_type<scalar_t, ROWS, COLS>& m, std::size_t row,
    std::size_t col) {

  return cmath::vector_getter<std::size_t, Vc::array, scalar_t, SIZE,
                              vc::storage_type<scalar_t, SIZE>>()(m, row, col);
}

/// @name Getter functions on @c algebra::vc::matrix_type
/// @{

using cmath::element;

/// @}

}  // namespace getter

namespace vector {

/// @name Vector functions on @c algebra::vc::storage_type
/// @{

using cmath::cross;
using cmath::dot;
using cmath::normalize;

using vc::math::cross;
using vc::math::dot;
using vc::math::normalize;

/// @}

}  // namespace vector

namespace matrix {

template <typename T, std::size_t ROWS, std::size_t COLS>
using matrix_type = vc::matrix_type<T, ROWS, COLS>;

template <typename size_type, typename scalar_t>
using element_getter_type =
    cmath::element_getter<size_type, Vc::array, scalar_t>;

// matrix actor
template <typename size_type, typename scalar_t, typename determinant_actor_t,
          typename inverse_actor_t>
using actor = cmath::matrix::actor<size_type, matrix_type, scalar_t,
                                   determinant_actor_t, inverse_actor_t,
                                   element_getter_type<size_type, scalar_t>>;

namespace determinant {

// determinant aggregation
template <typename size_type, typename scalar_t, class... As>
using actor =
    cmath::matrix::determinant::actor<size_type, matrix_type, scalar_t, As...>;

// determinant::cofactor
template <typename size_type, typename scalar_t, size_type... Ds>
using cofactor = cmath::matrix::determinant::cofactor<
    size_type, matrix_type, scalar_t, element_getter_type<size_type, scalar_t>,
    Ds...>;

// determinant::hard_coded
template <typename size_type, typename scalar_t, size_type... Ds>
using hard_coded = cmath::matrix::determinant::hard_coded<
    size_type, matrix_type, scalar_t, element_getter_type<size_type, scalar_t>,
    Ds...>;

// preset(s) as standard option(s) for user's convenience
template <typename size_type, typename scalar_t>
using preset0 = actor<size_type, scalar_t, cofactor<size_type, scalar_t>,
                      hard_coded<size_type, scalar_t, 2>>;

}  // namespace determinant

namespace inverse {

// inverion aggregation
template <typename size_type, typename scalar_t, class... As>
using actor =
    cmath::matrix::inverse::actor<size_type, matrix_type, scalar_t, As...>;

// inverse::cofactor
template <typename size_type, typename scalar_t, size_type... Ds>
using cofactor =
    cmath::matrix::inverse::cofactor<size_type, matrix_type, scalar_t,
                                     element_getter_type<size_type, scalar_t>,
                                     Ds...>;

// inverse::hard_coded
template <typename size_type, typename scalar_t, size_type... Ds>
using hard_coded =
    cmath::matrix::inverse::hard_coded<size_type, matrix_type, scalar_t,
                                       element_getter_type<size_type, scalar_t>,
                                       Ds...>;

// preset(s) as standard option(s) for user's convenience
template <typename size_type, typename scalar_t>
using preset0 = actor<size_type, scalar_t, cofactor<size_type, scalar_t>,
                      hard_coded<size_type, scalar_t, 2>>;

}  // namespace inverse

}  // namespace matrix

}  // namespace algebra
