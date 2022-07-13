// RUN: mlir-opt -test-tiling-interface=tile-consumer-and-fuse-producer-using-scf-for %s

func.func @conv_pool_fusion(
    %arg0 : tensor<?x?x?x?xf32>,
    %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?x?x?xf32>,
    %arg3 : tensor<?x?x?x?xf32>,
    %arg4 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_sum
      ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = linalg.conv_2d_nhwc_hwcf
      {__internal_linalg_transform__ = "conv_pool_fuse"}
      ins(%0, %arg3 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg4 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}