#include "stage_registry.cuh"

std::vector<StageDefinition> getRegisteredStages() {
  return {
      {"naive",
       "Uncoalesced SGEMM: threadIdx.x drives the row so a warp strides through A and C.",
       false, 1.0e-4f, launchNaiveSgemm, nullptr, describeNaiveSgemm},
      {"coalesced",
       "Coalesced SGEMM: threadIdx.x drives the column so a warp hits contiguous B and C.",
       false, 1.0e-4f, launchCoalescedSgemm, nullptr, describeCoalescedSgemm},
      {"tiled",
       "Shared-memory tiled SGEMM: 32x32 A/B tiles reused across a 32x32 C tile.",
       false, 1.0e-4f, launchTiledSgemm, nullptr, describeTiledSgemm},
      {"register",
       "Register-blocked SGEMM: 64x64 block tile, each thread owns a 4x4 C patch.",
       false, 1.0e-4f, launchRegisterBlockedSgemm, nullptr, describeRegisterBlockedSgemm},
      {"vectorized",
       "Register-blocked SGEMM with float4 global-to-shared loads.",
       false, 1.0e-4f, launchVectorizedSgemm, nullptr, describeVectorizedSgemm},
      {"wmma",
       "WMMA Tensor Core SGEMM: FP16 in, FP32 accumulate, 16x16x16 fragments.",
       true, 1.0e-2f, nullptr, launchWmmaSgemm, describeWmmaSgemm},
  };
}
