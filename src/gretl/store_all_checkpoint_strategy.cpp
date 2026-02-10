// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "store_all_checkpoint_strategy.hpp"
#include <cassert>
#include <iostream>

namespace gretl {

size_t StoreAllCheckpointStrategy::add_checkpoint_and_get_index_to_remove(size_t step, bool persistent)
{
  steps_.insert(step);
  if (persistent) {
    persistentSteps_.insert(step);
  }
  metrics_.stores++;
  return invalidCheckpointIndex;  // never evict
}

size_t StoreAllCheckpointStrategy::last_checkpoint_step() const
{
  assert(!steps_.empty());
  return *steps_.rbegin();
}

bool StoreAllCheckpointStrategy::erase_step(size_t stepIndex)
{
  if (persistentSteps_.count(stepIndex)) {
    return false;
  }
  return steps_.erase(stepIndex) > 0;
}

bool StoreAllCheckpointStrategy::contains_step(size_t stepIndex) const { return steps_.count(stepIndex) > 0; }

void StoreAllCheckpointStrategy::reset()
{
  for (auto it = steps_.begin(); it != steps_.end();) {
    if (persistentSteps_.count(*it) == 0) {
      it = steps_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t StoreAllCheckpointStrategy::capacity() const { return steps_.size(); }

size_t StoreAllCheckpointStrategy::size() const { return steps_.size(); }

void StoreAllCheckpointStrategy::print(std::ostream& os) const
{
  os << "CHECKPOINTS (StoreAll): size = " << steps_.size() << std::endl;
  for (const auto& s : steps_) {
    os << "   step=" << s << (persistentSteps_.count(s) ? " (persistent)" : "") << "\n";
  }
}

CheckpointMetrics StoreAllCheckpointStrategy::metrics() const { return metrics_; }

void StoreAllCheckpointStrategy::reset_metrics() { metrics_ = {}; }

void StoreAllCheckpointStrategy::record_recomputation() { metrics_.recomputations++; }

}  // namespace gretl
