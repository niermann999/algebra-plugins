/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/vc_vc.hpp"
#include "benchmark/common/benchmark_vector.hpp"
#include "benchmark/vc_aos/data_generator.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

using namespace algebra;

/// Run vector benchmarks
int main(int argc, char** argv) {

  constexpr std::size_t n_samples{160000};

  //
  // Prepare benchmarks
  //
  algebra::benchmark_base::configuration cfg{};
  cfg.n_samples(n_samples).n_warmup(
      static_cast<std::size_t>(0.1 * cfg.n_samples()));
  cfg.do_sleep(false);

  vector_binaryOP_bm<vc::vector3, float, bench_op::add> v_add_s{cfg};
  vector_binaryOP_bm<vc::vector3, float, bench_op::sub> v_sub_s{cfg};
  vector_binaryOP_bm<vc::vector3, float, bench_op::dot> v_dot_s{cfg};
  vector_binaryOP_bm<vc::vector3, float, bench_op::cross> v_cross_s{cfg};
  vector_unaryOP_bm<vc::vector3, float, bench_op::normalize> v_normalize_s{cfg};

  vector_binaryOP_bm<vc::vector3, double, bench_op::add> v_add_d{cfg};
  vector_binaryOP_bm<vc::vector3, double, bench_op::sub> v_sub_d{cfg};
  vector_binaryOP_bm<vc::vector3, double, bench_op::dot> v_dot_d{cfg};
  vector_binaryOP_bm<vc::vector3, double, bench_op::cross> v_cross_d{cfg};
  vector_unaryOP_bm<vc::vector3, double, bench_op::normalize> v_normalize_d{
      cfg};

  std::cout << "Algebra-Plugins 'vector' benchmark (Vc SoA)\n"
            << "-------------------------------------------\n\n"
            << cfg;

  //
  // Register all benchmarks
  //
  ::benchmark::RegisterBenchmark((v_add_s.name() + "_single").c_str(), v_add_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_add_d.name() + "_double").c_str(), v_add_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_sub_s.name() + "_single").c_str(), v_sub_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_sub_d.name() + "_double").c_str(), v_sub_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_dot_s.name() + "_single").c_str(), v_dot_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_dot_d.name() + "_double").c_str(), v_dot_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_cross_s.name() + "_single").c_str(),
                                 v_cross_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_cross_d.name() + "_double").c_str(),
                                 v_cross_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_normalize_s.name() + "_single").c_str(),
                                 v_normalize_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_normalize_d.name() + "_double").c_str(),
                                 v_normalize_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
}
