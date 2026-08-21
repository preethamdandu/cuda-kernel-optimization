#pragma once

#include <vector>

using SgemmLauncher = void (*)(const float*, const float*, float*, int, int, int);

struct StageDefinition {
  const char* name;
  const char* description;
  bool use_tensor_core_baseline;
  float rel_tolerance;
  SgemmLauncher launcher;
};

void launchNaiveSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchCoalescedSgemm(const float* a, const float* b, float* c, int m, int n, int k);

std::vector<StageDefinition> getRegisteredStages();
