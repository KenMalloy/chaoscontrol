#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shm_ring.h"
#include "wire_events.h"

uint64_t controller_main(
    const std::vector<std::string>& write_ring_names,
    const std::string& query_ring_name,
    const std::string& replay_ring_name,
    const std::string& exit_flag_shm_name,
    uint32_t idle_sleep_ns = 100);
