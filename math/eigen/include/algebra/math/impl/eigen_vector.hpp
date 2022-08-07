/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// Eigen include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Eigen/Core>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

namespace algebra::eigen::vector {

struct actor {

  /** This method retrieves phi from a vector, vector base with rows >= 2
   *
   * @param v the input vector
   **/
  template <
      typename derived_type,
      std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 2,
                       bool> = true>
  ALGEBRA_HOST_DEVICE inline auto phi(
      const Eigen::MatrixBase<derived_type> &v) noexcept {

    return algebra::math::atan2(v[1], v[0]);
  }

  /** This method retrieves theta from a vector, vector base with rows >= 3
   *
   * @param v the input vector
   **/
  template <
      typename derived_type,
      std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 3,
                       bool> = true>
  ALGEBRA_HOST_DEVICE inline auto theta(
      const Eigen::MatrixBase<derived_type> &v) noexcept {

    return algebra::math::atan2(algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]),
                                v[2]);
  }

  /** This method retrieves the perpenticular magnitude of a vector with rows >=
   *2
   *
   * @param v the input vector
   **/
  template <
      typename derived_type,
      std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 2,
                       bool> = true>
  ALGEBRA_HOST_DEVICE inline auto perp(
      const Eigen::MatrixBase<derived_type> &v) noexcept {

    return algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]);
  }

  /** This method retrieves the norm of a vector, no dimension restriction
   *
   * @param v the input vector
   **/
  template <typename derived_type>
  ALGEBRA_HOST_DEVICE inline auto norm(
      const Eigen::MatrixBase<derived_type> &v) {

    return v.norm();
  }

  /** This method retrieves the pseudo-rapidity from a vector or vector base
   *with rows >= 3
   *
   * @param v the input vector
   **/
  template <
      typename derived_type,
      std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 3,
                       bool> = true>
  ALGEBRA_HOST_DEVICE inline auto eta(
      const Eigen::MatrixBase<derived_type> &v) noexcept {

    return algebra::math::atanh(v[2] / v.norm());
  }

  /** Get a normalized version of the input vector
   *
   * @tparam derived_type is the matrix template
   *
   * @param v the input vector
   **/
  template <typename derived_type>
  ALGEBRA_HOST_DEVICE inline auto normalize(
      const Eigen::MatrixBase<derived_type> &v) {
    return v.normalized();
  }

  /** Dot product between two input vectors
   *
   * @tparam derived_type_lhs is the first matrix (epresseion) template
   * @tparam derived_type_rhs is the second matrix (epresseion) template
   *
   * @param a the first input vector
   * @param b the second input vector
   *
   * @return the scalar dot product value
   **/
  template <typename derived_type_lhs, typename derived_type_rhs>
  ALGEBRA_HOST_DEVICE inline auto dot(
      const Eigen::MatrixBase<derived_type_lhs> &a,
      const Eigen::MatrixBase<derived_type_rhs> &b) {
    return a.dot(b);
  }

  /** Cross product between two input vectors
   *
   * @tparam derived_type_lhs is the first matrix (epresseion) template
   * @tparam derived_type_rhs is the second matrix (epresseion) template
   *
   * @param a the first input vector
   * @param b the second input vector
   *
   * @return a vector (expression) representing the cross product
   **/
  template <typename derived_type_lhs, typename derived_type_rhs>
  ALGEBRA_HOST_DEVICE inline auto cross(
      const Eigen::MatrixBase<derived_type_lhs> &a,
      const Eigen::MatrixBase<derived_type_rhs> &b) {
    return a.cross(b);
  }
};

}  // namespace algebra::eigen::vector
