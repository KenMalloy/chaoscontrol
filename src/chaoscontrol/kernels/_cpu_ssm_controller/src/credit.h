#pragma once

#include <cstdint>

float recency_decay(float reward, uint64_t T, uint64_t P, float gamma = 0.995f);
