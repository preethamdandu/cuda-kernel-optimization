#pragma once

void launchUnfusedAttention(const float* q,
                            const float* k,
                            const float* v,
                            float* scores,
                            float* out,
                            int seq,
                            int head_dim);

void launchFusedAttention(const float* q,
                          const float* k,
                          const float* v,
                          float* out,
                          int seq,
                          int head_dim);
