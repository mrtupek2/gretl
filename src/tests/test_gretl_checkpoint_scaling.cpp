// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/// @file test_gretl_checkpoint_scaling.cpp
/// @brief Scaling study: fixed memory budget of 1000 checkpoints,
///        chain lengths from 1000 to 50000.  Outputs CSV for plotting.

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include "gtest/gtest.h"
#include "gretl/checkpoint_strategy.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/strumm_walther_checkpoint_strategy.hpp"
#include "gretl/store_all_checkpoint_strategy.hpp"
#include "gretl/periodic_checkpoint_strategy.hpp"
#include "gretl/state.hpp"
#include "gretl/data_store.hpp"

namespace {

gretl::State<double> forward_step_state(const gretl::State<double>& a)
{
  auto b = a.clone({a});

  b.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    downstream.set(upstreams[0].get<double>() / 3.0 + 2.0);
  });

  b.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    upstreams[0].get_dual<double, double>() += downstream.get_dual<double>() / 3.0;
  });

  return b.finalize();
}

struct ScalingResult {
  std::string strategy;
  size_t N;
  size_t budget_or_period;
  gretl::CheckpointMetrics metrics;
  double gradient;
  size_t peak_stored;
  double wall_seconds;
};

ScalingResult run_scaling_test(std::unique_ptr<gretl::CheckpointStrategy> strategy, const std::string& name, size_t N,
                               size_t budget_or_period)
{
  auto t0 = std::chrono::high_resolution_clock::now();

  gretl::DataStore dataStore(std::move(strategy));
  gretl::State<double> X = dataStore.create_state<double, double>(0.0);

  size_t peak = 0;
  for (size_t n = 0; n < N; ++n) {
    X = forward_step_state(X);
    size_t cur = dataStore.checkpointStrategy_->size();
    if (cur > peak) peak = cur;
  }

  X = set_as_objective(X);
  dataStore.stillConstructingGraph_ = false;
  dataStore.back_prop();

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  double grad = dataStore.get_dual<double, double>(0);
  return {name, N, budget_or_period, dataStore.checkpointStrategy_->metrics(), grad, peak, elapsed};
}

}  // namespace

TEST(CheckpointScaling, MemoryBudget1000)
{
  constexpr size_t BUDGET = 1000;
  std::vector<size_t> lengths = {1000, 2000, 5000, 10000, 20000, 50000};

  std::ofstream csv("checkpoint_scaling.csv");
  csv << "N,strategy,budget_or_period,stores,evictions,recomputations,peak_stored,wall_seconds\n";

  std::cout << "\n--- Checkpoint Scaling: Fixed Budget = " << BUDGET << " ---\n";
  std::cout << std::setw(8) << "N" << " | " << std::setw(14) << "Strategy" << std::setw(10) << "param" << std::setw(10)
            << "stores" << std::setw(10) << "evicts" << std::setw(12) << "recomps" << std::setw(10) << "peak_mem"
            << std::setw(12) << "ratio(r/N)" << std::setw(10) << "time(s)"
            << "\n";
  std::cout << std::string(98, '-') << "\n";

  for (size_t N : lengths) {
    std::vector<ScalingResult> results;

    // StoreAll — unlimited memory baseline (recomps = 0, memory = N)
    results.push_back(
        run_scaling_test(std::make_unique<gretl::StoreAllCheckpointStrategy>(), "StoreAll", N, N));

    // Wang — bounded memory
    results.push_back(
        run_scaling_test(std::make_unique<gretl::WangCheckpointStrategy>(BUDGET), "Wang", N, BUDGET));

    // StrummWalther — bounded memory
    results.push_back(
        run_scaling_test(std::make_unique<gretl::StrummWaltherCheckpointStrategy>(BUDGET), "StrummWalther", N, BUDGET));

    // Periodic — choose period so peak stored ≈ BUDGET
    size_t period = std::max(size_t(1), (N + BUDGET - 1) / BUDGET);
    results.push_back(
        run_scaling_test(std::make_unique<gretl::PeriodicCheckpointStrategy>(period), "Periodic", N, period));

    // Verify all strategies produce the same gradient
    double ref_grad = results[0].gradient;
    for (const auto& r : results) {
      ASSERT_NEAR(r.gradient, ref_grad, 1e-10) << r.strategy << " gradient mismatch at N=" << N;
    }

    for (const auto& r : results) {
      csv << r.N << "," << r.strategy << "," << r.budget_or_period << "," << r.metrics.stores << ","
          << r.metrics.evictions << "," << r.metrics.recomputations << "," << r.peak_stored << "," << std::fixed
          << std::setprecision(4) << r.wall_seconds << "\n";

      std::cout << std::setw(8) << r.N << " | " << std::setw(14) << r.strategy << std::setw(10) << r.budget_or_period
                << std::setw(10) << r.metrics.stores << std::setw(10) << r.metrics.evictions << std::setw(12)
                << r.metrics.recomputations << std::setw(10) << r.peak_stored << std::setw(12) << std::fixed
                << std::setprecision(3) << static_cast<double>(r.metrics.recomputations) / static_cast<double>(N)
                << std::setw(10) << std::setprecision(3) << r.wall_seconds << "\n";
    }
  }

  csv.close();
  std::cout << "\nResults written to checkpoint_scaling.csv\n" << std::endl;
}
