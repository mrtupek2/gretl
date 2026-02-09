// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file online_r2_checkpoint_strategy.hpp
 * @brief Stumm & Walther 2010 "Online r=2" checkpointing strategy.
 *
 * Reference: Philipp Stumm and Andrea Walther, "New Algorithms for Optimal
 * Online Checkpointing", SIAM J. Sci. Comput., 32(2), 836-854, 2010.
 * DOI: 10.1137/080742439
 */

#pragma once

#include "checkpoint_strategy.hpp"
#include <vector>
#include <algorithm>

namespace gretl {

/// @brief Stumm & Walther 2010 "Online r=2" checkpointing strategy.
///
/// Unlike the Wang algorithm which uses levels to determine dispensability,
/// this algorithm maintains checkpoints with approximately uniform spacing
/// relative to the current step count. When at capacity, the eviction
/// candidate is the non-persistent checkpoint whose removal results in the
/// smallest maximum gap between remaining checkpoints.
///
/// Key properties:
/// - No level concept; eviction is based on spacing analysis
/// - Works online: total number of steps need not be known a priori
/// - Achieves near-optimal checkpoint distribution for unknown-length runs
class OnlineR2CheckpointStrategy final : public CheckpointStrategy {
 public:
  /// @brief Construct with a given number of non-persistent checkpoint slots.
  explicit OnlineR2CheckpointStrategy(size_t maxStates);

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
  /// @brief A checkpoint slot: stores step, persistent flag, and weight.
  struct Slot {
    size_t step;
    bool persistent;
    size_t weight;  ///< Importance weight; increases via promotion (like Wang levels)
  };

  /// @brief Find a "dispensable" slot using weight-based priority.
  /// Iterates from highest to lowest step; a slot is dispensable if its
  /// weight is less than the running maximum weight seen so far.
  /// @return Index of the dispensable slot, or slots_.size() if none found.
  size_t find_dispensable() const;

  /// @brief Find the index of the rightmost non-persistent slot.
  size_t find_rightmost_nonpersistent() const;

  size_t maxNumSlots_;
  std::vector<Slot> slots_;  ///< Sorted by step number
  CheckpointMetrics metrics_;
};

}  // namespace gretl
