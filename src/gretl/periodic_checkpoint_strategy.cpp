// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "periodic_checkpoint_strategy.hpp"
#include <cassert>
#include <iostream>

namespace gretl {

PeriodicCheckpointStrategy::PeriodicCheckpointStrategy(size_t period) : period_(period) { assert(period_ >= 1); }

size_t PeriodicCheckpointStrategy::add_checkpoint_and_get_index_to_remove(size_t step, bool persistent)
{
  size_t evictStep = invalidCheckpointIndex;

  if (persistent) {
    steps_.insert(step);
    persistentSteps_.insert(step);
    metrics_.stores++;
    return invalidCheckpointIndex;
  }

  // Periodic steps (multiples of period_) are always retained.
  // Non-periodic steps occupy a single working slot — evict the
  // previous non-periodic, non-persistent step when a new one arrives.
  if (!is_periodic(step)) {
    // Find the most recent non-periodic, non-persistent step to evict
    for (auto it = steps_.rbegin(); it != steps_.rend(); ++it) {
      size_t s = *it;
      if (!is_periodic(s) && persistentSteps_.count(s) == 0) {
        evictStep = s;
        steps_.erase(s);
        break;
      }
    }
  }

  steps_.insert(step);
  metrics_.stores++;
  if (valid_checkpoint_index(evictStep)) {
    metrics_.evictions++;
  }
  return evictStep;
}

size_t PeriodicCheckpointStrategy::last_checkpoint_step() const
{
  assert(!steps_.empty());
  return *steps_.rbegin();
}

bool PeriodicCheckpointStrategy::erase_step(size_t stepIndex)
{
  if (persistentSteps_.count(stepIndex)) {
    return false;
  }
  return steps_.erase(stepIndex) > 0;
}

bool PeriodicCheckpointStrategy::contains_step(size_t stepIndex) const { return steps_.count(stepIndex) > 0; }

void PeriodicCheckpointStrategy::reset()
{
  for (auto it = steps_.begin(); it != steps_.end();) {
    if (persistentSteps_.count(*it) == 0) {
      it = steps_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t PeriodicCheckpointStrategy::capacity() const { return period_; }

size_t PeriodicCheckpointStrategy::size() const { return steps_.size(); }

void PeriodicCheckpointStrategy::print(std::ostream& os) const
{
  os << "CHECKPOINTS (Periodic): period = " << period_ << ", size = " << steps_.size() << std::endl;
  for (const auto& s : steps_) {
    os << "   step=" << s;
    if (persistentSteps_.count(s)) {
      os << " (persistent)";
    } else if (is_periodic(s)) {
      os << " (periodic)";
    }
    os << "\n";
  }
}

CheckpointMetrics PeriodicCheckpointStrategy::metrics() const { return metrics_; }

void PeriodicCheckpointStrategy::reset_metrics() { metrics_ = {}; }

void PeriodicCheckpointStrategy::record_recomputation() { metrics_.recomputations++; }

}  // namespace gretl
