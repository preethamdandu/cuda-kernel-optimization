#pragma once

#include "kernel_occupancy.cuh"

#include <vector>

void launchUnfusedAttention(const float* q,
                            const float* k,
                            const float* v,
                            float* scores,
                            float* out,
                            int seq,
                            int head_dim);

// v1: one thread per Q row, 32-thread blocks. Kept so the v1-vs-v2 occupancy
// difference stays measurable rather than just described.
void launchFusedAttention(const float* q,
                          const float* k,
                          const float* v,
                          float* out,
                          int seq,
                          int head_dim);

// v2: one warp per Q row, 128-thread blocks.
void launchFusedAttentionV2(const float* q,
                            const float* k,
                            const float* v,
                            float* out,
                            int seq,
                            int head_dim);

// Each attention kernel lives in an anonymous namespace in its own translation
// unit, so the reporting has to happen where the symbol is visible. seq is
// needed to reconstruct the grid.
std::vector<KernelOccupancy> describeUnfusedAttention(int seq, int head_dim);
std::vector<KernelOccupancy> describeFusedAttention(int seq);
std::vector<KernelOccupancy> describeFusedAttentionV2(int seq);
