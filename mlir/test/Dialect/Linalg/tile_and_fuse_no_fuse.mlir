// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.generic fuse tile-sizes=0,0 run-enable-pass=false" -cse -split-input-file | FileCheck %s

builtin.func @gemm_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg3 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg4: f32, %arg5 : f32, %arg6 : f32):
        %1 = arith.addf %arg4, %arg5 : f32
        linalg.yield %1 : f32
      } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @gemm_bias_add(
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.matmul
//       CHECK:   linalg.generic
