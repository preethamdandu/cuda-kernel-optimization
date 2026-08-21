#include "stage_registry.cuh"

std::vector<StageDefinition> getRegisteredStages() {
  return {
      {
          "naive",
          "Uncoalesced SGEMM: threadIdx.x drives the row so a warp strides through A and C.",
          false,
          1.0e-4f,
          launchNaiveSgemm,
      },
      {
          "coalesced",
          "Coalesced SGEMM: threadIdx.x drives the column so a warp hits contiguous B and C.",
          false,
          1.0e-4f,
          launchCoalescedSgemm,
      },
  };
}
