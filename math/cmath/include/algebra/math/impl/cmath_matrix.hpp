/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/algorithms/utils/algorithm_finder.hpp"
#include "algebra/qualifiers.hpp"

namespace algebra::cmath::matrix {

/// "Matrix actor", assuming a simple 2D matrix
template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class determinant_actor_t, class inverse_actor_t,
          class element_getter_t>
struct actor {

  /// Function (object) used for accessing a matrix element
  using element_getter = element_getter_t;

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  // Create zero matrix
  template <size_type ROWS, size_type COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> zero() {
    matrix_type<ROWS, COLS> ret;

    for (size_type i = 0; i < ROWS; ++i) {
      for (size_type j = 0; j < COLS; ++j) {
        element_getter()(ret, i, j) = 0;
      }
    }

    return ret;
  }

  // Create identity matrix
  template <size_type ROWS, size_type COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> identity() {
    matrix_type<ROWS, COLS> ret;

    for (size_type i = 0; i < ROWS; ++i) {
      for (size_type j = 0; j < COLS; ++j) {
        if (i == j) {
          element_getter()(ret, i, j) = 1;
        } else {
          element_getter()(ret, i, j) = 0;
        }
      }
    }

    return ret;
  }

  // Create transpose matrix
  template <size_type ROWS, size_type COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<COLS, ROWS> transpose(
      const matrix_type<ROWS, COLS> &m) {

    matrix_type<COLS, ROWS> ret;

    for (size_type i = 0; i < ROWS; ++i) {
      for (size_type j = 0; j < COLS; ++j) {
        element_getter()(ret, j, i) = element_getter()(m, i, j);
      }
    }

    return ret;
  }

  // Get determinant
  template <size_type N>
  ALGEBRA_HOST_DEVICE inline scalar_t determinant(const matrix_type<N, N> &m) {

    return determinant_actor_t()(m);
  }

  // Create inverse matrix
  template <size_type N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> inverse(
      const matrix_type<N, N> &m) {

    return inverse_actor_t()(m);
  }
};

namespace determinant {

template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class... As>
struct actor {

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline scalar_t operator()(
      const matrix_type<N, N> &m) const {

    return typename find_algorithm<size_type, N, As...>::algorithm_type()(m);
  }
};

}  // namespace determinant

namespace inverse {

template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class... As>
struct actor {

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
      const matrix_type<N, N> &m) const {

    return typename find_algorithm<size_type, N, As...>::algorithm_type()(m);
  }
};

}  // namespace inverse

}  // namespace algebra::cmath::matrix