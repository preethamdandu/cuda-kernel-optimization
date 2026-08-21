#pragma once

#include <cuda_fp16.h>
#include <vector>

using SgemmLauncher = void (*)(const float*, const float*, float*, int, int, int);
using WmmaLauncher = void (*)(const __half*, const __half*, float*, int, int, int);

struct StageDefinition {
  const char* name;
  const char* description;
  bool use_tensor_core_baseline;
  float rel_tolerance;
  SgemmLauncher launcher;
  WmmaLauncher wmma_launcher;
};

void launchNaiveSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchCoalescedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchTiledSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchRegisterBlockedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchVectorizedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchWmmaSgemm(const __half* a, const __half* b, float* c, int m, int n, int k);

std::vector<StageDefinition> getRegisteredStages();
