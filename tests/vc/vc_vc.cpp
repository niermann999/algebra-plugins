/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "algebra/vc_vc.hpp"

// Test include(s).
#include "test_host_basics.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <string>

/// Struct providing a readable name for the test
struct test_specialisation_name {
  template <typename T>
  static std::string GetName(int i) {
    switch (i) {
      case 0:
        return "vc_vc<float>";
      case 1:
        return "vc_vc<double>";
      default:
        return "unknown";
    }
  }
};

// Instantiate the test(s).
typedef testing::Types<
    test_types<float, std::array<float, 2>, std::array<float, 3>,
               std::array<float, 2>, std::array<float, 3>,
               algebra::vc::transform3<float>, std::size_t,
               algebra::vc::matrix_type,
               algebra::matrix::actor<
                   float, algebra::matrix::determinant::preset0<float>,
                   algebra::matrix::inverse::preset0<float>>>,
    test_types<double, std::array<double, 2>, std::array<double, 3>,
               std::array<double, 2>, std::array<double, 3>,
               algebra::vc::transform3<double>, std::size_t,
               algebra::vc::matrix_type,
               algebra::matrix::actor<
                   double, algebra::matrix::determinant::preset0<double>,
                   algebra::matrix::inverse::preset0<double>>>>
    vc_vc_types;
INSTANTIATE_TYPED_TEST_SUITE_P(algebra_plugins, test_host_basics, vc_vc_types,
                               test_specialisation_name);