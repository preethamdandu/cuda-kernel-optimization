#pragma once

#include "kernel_occupancy.cuh"

#include <cuda_fp16.h>
#include <vector>

using SgemmLauncher = void (*)(const float*, const float*, float*, int, int, int);
using WmmaLauncher = void (*)(const __half*, const __half*, float*, int, int, int);
// Each kernel is in an anonymous namespace in its own translation unit, so
// occupancy has to be reported from there rather than from the harness.
using StageDescriber = std::vector<KernelOccupancy> (*)(int);

struct StageDefinition {
  const char* name;
  const char* description;
  bool use_tensor_core_baseline;
  float rel_tolerance;
  SgemmLauncher launcher;
  WmmaLauncher wmma_launcher;
  StageDescriber describer;
};

void launchNaiveSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchCoalescedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchTiledSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchRegisterBlockedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchVectorizedSgemm(const float* a, const float* b, float* c, int m, int n, int k);
void launchWmmaSgemm(const __half* a, const __half* b, float* c, int m, int n, int k);

std::vector<KernelOccupancy> describeNaiveSgemm(int n);
std::vector<KernelOccupancy> describeCoalescedSgemm(int n);
std::vector<KernelOccupancy> describeTiledSgemm(int n);
std::vector<KernelOccupancy> describeRegisterBlockedSgemm(int n);
std::vector<KernelOccupancy> describeVectorizedSgemm(int n);
std::vector<KernelOccupancy> describeWmmaSgemm(int n);

std::vector<StageDefinition> getRegisteredStages();
