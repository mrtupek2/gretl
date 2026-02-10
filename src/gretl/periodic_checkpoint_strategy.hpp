// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file periodic_checkpoint_strategy.hpp
 * @brief Checkpoint strategy that retains every Pth step and recomputes
 *        in between.
 *
 * Memory scales as O(N/P) where N is the total number of steps and P is
 * the period.  During the reverse pass, up to P-1 forward re-evaluations
 * are needed per block, giving O(N) total recomputation — the same
 * asymptotic cost as a single forward pass.
 */

#pragma once

#include "checkpoint_strategy.hpp"
#include <set>

namespace gretl {

/// @brief Checkpoint strategy that stores every Pth step.
///
/// Steps whose index is a multiple of the period are retained until
/// explicitly erased during the reverse pass.  Non-periodic steps are
/// kept in a single working slot and evicted as soon as the next
/// non-periodic step arrives.
class PeriodicCheckpointStrategy final : public CheckpointStrategy {
 public:
  /// @brief Construct with a given checkpoint period.
  /// @param period Store a checkpoint every `period` steps (must be >= 1).
  explicit PeriodicCheckpointStrategy(size_t period);

  size_t add_checkpoint_and_get_index_to_remove(size_t step, bool persistent = false) override;
  size_t last_checkpoint_step() const override;
  bool erase_step(size_t stepIndex) override;
  bool contains_step(size_t stepIndex) const override;
  void reset() override;
  size_t capacity() const override;
  size_t size() const override;
  void print(std::ostream& os) const override;
  CheckpointMetrics metrics() const override;
  void reset_metrics() override;
  void record_recomputation() override;

 private:
  bool is_periodic(size_t step) const { return step % period_ == 0; }

  size_t period_;
  std::set<size_t> steps_;             ///< All currently stored steps
  std::set<size_t> persistentSteps_;   ///< Steps marked persistent (graph roots)
  CheckpointMetrics metrics_;
};

}  // namespace gretl
