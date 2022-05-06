// RUN: mlir-opt %s  -test-linalg-tile-and-fuse-on-tensors -cse --split-input-file | FileCheck %s

func.func @matmul_bias_add(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>, %bias : tensor<?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      __internal_linalg_transform__ = "return_fused_values"}
      ins(%matmul, %bias : tensor<?x?xf32>, tensor<?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b3 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
    } -> tensor<?x?xf32>
  return %matmul, %bias_add : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_bias_add
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step
//  CHECK-SAME:       iter_args(%[[OUTER_ITER0:[a-zA-Z0-9]+]] = %[[INIT]],
//  CHECK-SAME:       %[[OUTER_ITER1:[a-zA-Z0-9]+]] = %[[INIT]])
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:.+]] = %{{.+}} to %[[UB1:.+]] step
//  CHECK-SAME:       iter_args(%[[INNER_ITER0:[a-zA-Z0-9]+]] = %[[OUTER_ITER0]],
//  CHECK-SAME:       %[[INNER_ITER1:[a-zA-Z0-9]+]] = %[[OUTER_ITER1]])
//   CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//   CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[INNER_ITER1]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[FILL_TILE:.+]] = linalg.fill
//  CHECK-SAME:           outs(%[[OUTS_TILE]] :
//   CHECK-DAG:       %[[MATMUL_TILE:.+]] = linalg.matmul
//  CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
//  CHECK-SAME:           outs(%[[FILL_TILE:.+]] :
//   CHECK-DAG:       %[[BIAS_TILE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV1]]]
//   CHECK-DAG:       %[[OUTS2_TILE:.+]] = tensor.extract_slice %[[INNER_ITER0]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[ROOT_TILE:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[MATMUL_TILE]], %[[BIAS_TILE]] :
//  CHECK-SAME:           outs(%[[OUTS2_TILE]] :
//   CHECK-DAG:       %[[ROOT_INSERT:.+]] = tensor.insert_slice %[[ROOT_TILE]] into %[[INNER_ITER0]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[MATMUL_INSERT:.+]] = tensor.insert_slice %[[MATMUL_TILE]] into %[[INNER_ITER1]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[ROOT_INSERT]], %[[MATMUL_INSERT]]
//       CHECK:     scf.yield %[[INNER]]#0, %[[INNER]]#1
//       CHECK:   return %[[OUTER]]#1, %[[OUTER]]#0

// -----

func.func @matmul_lhs_fusion(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %AB_init: tensor<?x?xf32>, %C: tensor<?x?xf32>, %ABC_init: tensor<?x?xf32>)
     -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %AB = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%AB_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN1> <N1xN2>
  %ABC = linalg.matmul {__internal_linalg_transform__ = "lhs_fusion"}
    ins(%AB, %C : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%ABC_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN2> <N2xN3>
  return %AB, %ABC : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_lhs_fusion
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[AB_INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[C:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ABC_INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[LOOP:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]]
//  CHECK-SAME:       iter_args(%[[ITER0:[a-zA-Z0-9]+]] = %[[ABC_INIT]],
//  CHECK-SAME:       %[[ITER1:[a-zA-Z0-9]+]] = %[[AB_INIT]])
//   CHECK-DAG:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[A]][%[[IV0]], 0]
//   CHECK-DAG:     %[[OUTS0_TILE:.+]] = tensor.extract_slice %[[ITER1]][%[[IV0]], 0]
//   CHECK-DAG:     %[[GEMM0_TILE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[B]] :
//  CHECK-SAME:         outs(%[[OUTS0_TILE]] :
//   CHECK-DAG:     %[[OUTS1_TILE:.+]] = tensor.extract_slice %[[ITER0]][%[[IV0]], 0]
//   CHECK-DAG:     %[[GEMM1_TILE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[GEMM0_TILE]], %[[C]] :
//  CHECK-SAME:         outs(%[[OUTS1_TILE:.+]] :
//   CHECK-DAG:     %[[GEMM1_INSERT:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITER0]][%[[IV0]], 0]
//   CHECK-DAG:     %[[GEMM0_INSERT:.+]] = tensor.insert_slice %[[GEMM0_TILE]] into %[[ITER1]][%[[IV0]], 0]
//       CHECK:     scf.yield %[[GEMM1_INSERT]], %[[GEMM0_INSERT]]
//       CHECK:   return %[[LOOP]]#1, %[[LOOP]]#0

// -----

func.func @matmul_rhs_fusion(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %AB_init: tensor<?x?xf32>, %C: tensor<?x?xf32>, %ABC_init: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %AB = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%AB_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN1> <N1xN2>
  %ABC = linalg.matmul {__internal_linalg_transform__ = "rhs_fusion"}
    ins(%C, %AB : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%ABC_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN2> <N2xN3>
  return %AB, %ABC : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_rhs_fusion
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[AB_INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[C:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ABC_INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[LOOP:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]]
//  CHECK-SAME:       iter_args(%[[ITER0:[a-zA-Z0-9]+]] = %[[ABC_INIT]],
//  CHECK-SAME:       %[[ITER1:[a-zA-Z0-9]+]] = %[[AB_INIT]])
//   CHECK-DAG:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[B]][0, %[[IV0]]]
//   CHECK-DAG:     %[[OUTS0_TILE:.+]] = tensor.extract_slice %[[ITER1]][0, %[[IV0]]]
//   CHECK-DAG:     %[[GEMM0_TILE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[A]], %[[RHS_TILE]] :
//  CHECK-SAME:         outs(%[[OUTS0_TILE]] :
//   CHECK-DAG:     %[[OUTS1_TILE:.+]] = tensor.extract_slice %[[ITER0]][0, %[[IV0]]]
//   CHECK-DAG:     %[[GEMM1_TILE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[C]], %[[GEMM0_TILE]] :
//  CHECK-SAME:         outs(%[[OUTS1_TILE:.+]] :
//   CHECK-DAG:     %[[GEMM1_INSERT:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITER0]][0, %[[IV0]]]
//   CHECK-DAG:     %[[GEMM0_INSERT:.+]] = tensor.insert_slice %[[GEMM0_TILE]] into %[[ITER1]][0, %[[IV0]]]
//       CHECK:     scf.yield %[[GEMM1_INSERT]], %[[GEMM0_INSERT]]
//       CHECK:   return %[[LOOP]]#1, %[[LOOP]]#0

// -----

func.func @matmul_out_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul {__internal_linalg_transform__ = "matmul_outs_fusion"}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_out_fusion
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[OUTER:.+]] = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step
//  CHECK-SAME:       iter_args(%[[OUTER_ITER:[a-zA-Z0-9]+]] = %[[INIT]])
//       CHECK:     %[[INNER:.+]] = scf.for %[[IV1:.+]] = %{{.+}} to %[[UB1:.+]] step
//  CHECK-SAME:       iter_args(%[[INNER_ITER:[a-zA-Z0-9]+]] = %[[OUTER_ITER]])
//   CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//   CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[INNER_ITER]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[FILL_TILE:.+]] = linalg.fill
//  CHECK-SAME:           outs(%[[OUTS_TILE]] :
//   CHECK-DAG:       %[[MATMUL_TILE:.+]] = linalg.matmul
//  CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
//  CHECK-SAME:           outs(%[[FILL_TILE:.+]] :
//   CHECK-DAG:       %[[MATMUL_INSERT:.+]] = tensor.insert_slice %[[MATMUL_TILE]] into %[[INNER_ITER]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[MATMUL_INSERT]]
//       CHECK:     scf.yield %[[INNER]]
//       CHECK:   return %[[OUTER]]

// -----

func.func @generic_plus_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : tensor<?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %bias = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg2 : tensor<?xf32>) outs(%init: tensor<?x?xf32>) {
    ^bb(%b0: f32, %b1: f32) :
      linalg.yield %b0 : f32
    } -> tensor<?x?xf32>
  %gemm = linalg.matmul {__internal_linalg_transform__ = "matmul_outs_fusion"}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%bias : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %bias, %gemm : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @generic_plus_matmul
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step
//  CHECK-SAME:       iter_args(%[[OUTER_ITER0:[a-zA-Z0-9]+]] = %[[INIT]],
//  CHECK-SAME:       %[[OUTER_ITER1:[a-zA-Z0-9]+]] = %[[INIT]])
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:.+]] = %{{.+}} to %[[UB1:.+]] step
//  CHECK-SAME:       iter_args(%[[INNER_ITER0:[a-zA-Z0-9]+]] = %[[OUTER_ITER0]],
//  CHECK-SAME:       %[[INNER_ITER1:[a-zA-Z0-9]+]] = %[[OUTER_ITER1]])
//   CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//   CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[INNER_ITER0]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[BIAS_TILE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV1]]]
//       CHECK:       %[[BCAST_TILE:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[BIAS_TILE]] :
//  CHECK-SAME:           outs(%[[OUTS_TILE]] :
//   CHECK-DAG:       %[[MATMUL_TILE:.+]] = linalg.matmul
//  CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
//  CHECK-SAME:           outs(%[[BCAST_TILE:.+]] :
//   CHECK-DAG:       %[[MATMUL_INSERT:.+]] = tensor.insert_slice %[[MATMUL_TILE]] into %[[INNER_ITER0]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[BCAST_INSERT:.+]] = tensor.insert_slice %[[BCAST_TILE]] into %[[INNER_ITER1]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[MATMUL_INSERT]], %[[BCAST_INSERT]]
//       CHECK:     scf.yield %[[INNER]]#0, %[[INNER]]#1
//       CHECK:   return %[[OUTER]]#1, %[[OUTER]]#0
