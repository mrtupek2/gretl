// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file store_all_checkpoint_strategy.hpp
 * @brief Checkpoint strategy that stores every state without eviction.
 *
 * This is the simplest possible strategy: no algorithmic overhead, no
 * recomputation, but O(N) memory.  Useful as a baseline for comparing
 * the memory/compute trade-offs of smarter strategies.
 */

#pragma once

#include "checkpoint_strategy.hpp"
#include <set>

namespace gretl {

/// @brief Checkpoint strategy that retains every state — zero recomputation,
///        maximum memory.
class StoreAllCheckpointStrategy final : public CheckpointStrategy {
 public:
  StoreAllCheckpointStrategy() = default;

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
  std::set<size_t> steps_;             ///< All currently stored steps
  std::set<size_t> persistentSteps_;   ///< Subset that cannot be erased
  CheckpointMetrics metrics_;
};

}  // namespace gretl
