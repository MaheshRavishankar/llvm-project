// RUN: mlir-opt -test-tiling-interface=tile-consumer-and-fuse-yield-producer-using-scf-for -cse -split-input-file %s | FileCheck %s

func.func @gemm_generic_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %generic = linalg.generic {
      __internal_linalg_transform__ = "fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%gemm, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%init1 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32 
  } -> tensor<?x?xf32>
  return %gemm, %generic : tensor<?x?xf32>, tensor<?x?xf32> 
}
//      CHECK: func.func @gemm_generic_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>)
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor
//      CHECK:   %[[RESULT:[a-zA-Z0-9]+]]:3 = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:.+]] = %[[INIT]], %[[ITERARG1:.+]] = %[[INIT]], %[[ITERARG2:.+]] = %[[INIT]])
//      CHECK:     %[[INNER_RESULT:[a-zA-Z0-9]+]]:3 = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[INNER_ITERARG0:[a-zA-Z0-9]+]] = %[[ITERARG0]], %[[INNER_ITERARG1:[a-zA-Z0-9]+]] = %[[ITERARG1]], %[[INNER_ITERARG2:[a-zA-Z0-9]+]] = %[[ITERARG2]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = tensor.extract_slice %[[INNER_ITERARG2]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//  CHECK-DAG:       %[[BIAS_TILE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV1]]]
//  CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[INNER_ITERARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[GEMM_TILE]], %[[BIAS_TILE]] :
// CHECK-SAME:           outs(%[[OUTS_TILE]] :
//  CHECK-DAG:       %[[INSERT1:.+]] = tensor.insert_slice %[[GENERIC_TILE]] into %[[INNER_ITERARG0]][%[[IV0]], %[[IV1]]]
//  CHECK-DAG:       %[[INSERT2:.+]] = tensor.insert_slice %[[GEMM_TILE]] into %[[INNER_ITERARG1]][%[[IV0]], %[[IV1]]]
//  CHECK-DAG:       %[[INSERT3:.+]] = tensor.insert_slice %[[FILL_TILE]] into %[[INNER_ITERARG2]][%[[IV0]], %[[IV1]]]
//      CHECK:       scf.yield %[[INSERT1]], %[[INSERT2]], %[[INSERT3]]
//      CHECK:     scf.yield %[[INNER_RESULT]]#0, %[[INNER_RESULT]]#1, %[[INNER_RESULT]]#2
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @gemm_gemm_fusion(%lhs0 : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>, %rhs1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs0, %c1 : tensor<?x?xf32>
  %init0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill0 = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm0 = linalg.matmul
      ins(%lhs0, %rhs0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %d2 = tensor.dim %rhs1, %c1 : tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d0, %d2] : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm1 = linalg.matmul  {__internal_linalg_transform__ = "gemm_fusion"}
      ins(%gemm0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm0, %gemm1 : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func.func @gemm_gemm_fusion(
// CHECK-SAME:     %[[LHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[LHS0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[RHS0]], %[[C1]]
//  CHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[RHS1]], %[[C1]]
//      CHECK:   %[[INIT1:.+]] = linalg.init_tensor [%[[D0]], %[[D2]]]
//      CHECK:   %[[RESULT:[a-zA-Z0-9]+]]:4 = scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG2:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG3:[a-zA-Z0-9]+]] = %[[INIT1]])
//  CHECK-DAG:     %[[LHS0_TILE:.+]] = tensor.extract_slice %[[LHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[RHS0_TILE:.+]] = tensor.extract_slice %[[RHS0]][0, 0]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[ITERARG2]][%[[IV]], 0]
//      CHECK:     %[[FILL0_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT0_TILE]] :
//      CHECK:     %[[GEMM0_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS0_TILE]], %[[RHS0_TILE]] :
// CHECK-SAME:         outs(%[[FILL0_TILE]] :
//  CHECK-DAG:     %[[RHS1_TILE:.+]] = tensor.extract_slice %[[RHS1]][0, 0]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[ITERARG3]][%[[IV]], 0]
//      CHECK:     %[[FILL1_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT1_TILE]] :
//      CHECK:     %[[GEMM1_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[GEMM0_TILE]], %[[RHS1_TILE]] :
// CHECK-SAME:         outs(%[[FILL1_TILE]] :
//  CHECK-DAG:     %[[INSERT0:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITERARG0]][%[[IV]], 0]
//  CHECK-DAG:     %[[INSERT1:.+]] = tensor.insert_slice %[[GEMM0_TILE]] into %[[ITERARG1]][%[[IV]], 0]
//  CHECK-DAG:     %[[INSERT2:.+]] = tensor.insert_slice %[[FILL0_TILE]] into %[[ITERARG2]][%[[IV]], 0]
//  CHECK-DAG:     %[[INSERT3:.+]] = tensor.insert_slice %[[FILL1_TILE]] into %[[ITERARG3]][%[[IV]], 0]
//      CHECK:     scf.yield %[[INSERT0]], %[[INSERT1]], %[[INSERT2]], %[[INSERT3]]
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @interchange_matmul_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %4 = linalg.generic {
      __internal_linalg_transform__ = "gemm_interchange_fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%2 : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %4 = arith.addf %b0, %b0 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
  return %2, %4 : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func.func @interchange_matmul_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[INIT0:.+]] = linalg.init_tensor
//      CHECK:   %[[RESULT:[a-zA-Z0-9]+]]:3 = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT]], %[[ITERARG2:[a-zA-Z0-9]+]] = %[[INIT]])
//      CHECK:     %[[INNER_RESULT:[a-zA-Z0-9]+]]:3 = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[INNER_ITERARG0:[a-zA-Z0-9]+]] = %[[ITERARG0]], %[[INNER_ITERARG1:[a-zA-Z0-9]+]] = %[[ITERARG1]], %[[INNER_ITERARG2:[a-zA-Z0-9]+]] = %[[ITERARG2]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV1]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV0]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = tensor.extract_slice %[[INNER_ITERARG2]][%[[IV1]], %[[IV0]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//      CHECK:       %[[INIT_TILE_2:.+]] = tensor.extract_slice %[[INNER_ITERARG0]][%[[IV1]], %[[IV0]]]
//      CHECK:       %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[GEMM_TILE]] :
// CHECK-SAME:           outs(%[[INIT_TILE_2]] :
//  CHECK-DAG:       %[[INSERT0:.+]] = tensor.insert_slice %[[GENERIC_TILE]] into %[[INNER_ITERARG0]][%[[IV1]], %[[IV0]]]
//  CHECK-DAG:       %[[INSERT1:.+]] = tensor.insert_slice %[[GEMM_TILE]] into %[[INNER_ITERARG1]][%[[IV1]], %[[IV0]]]
//  CHECK-DAG:       %[[INSERT2:.+]] = tensor.insert_slice %[[FILL_TILE]] into %[[INNER_ITERARG2]][%[[IV1]], %[[IV0]]]
//      CHECK:       scf.yield %[[INSERT0]], %[[INSERT1]], %[[INSERT2]]
//      CHECK:     scf.yield %[[INNER_RESULT]]#0, %[[INNER_RESULT]]#1, %[[INNER_RESULT]]#2
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @matmul_sequence_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>,
    %arg5: tensor<?x?xf32>, %arg6: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul
    {__internal_linalg_transform__ = "gemm_sequence_fusion"}
    ins(%1, %arg5 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N2] * [N2, N3]
  return %0, %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @matmul_sequence_fusion(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[N0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[ORIG_GEMM1:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] :
//   CHECK-DAG:   %[[N1:.+]] = tensor.dim %[[ORIG_GEMM1]], %[[C1]]
//   CHECK-DAG:   %[[ORIG_GEMM2:.+]] = linalg.matmul ins(%[[ORIG_GEMM1]], %[[ARG3]] :
//   CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ORIG_GEMM2]], %[[C0]]
//   CHECK-DAG:   %[[N2:.+]] = tensor.dim %[[ORIG_GEMM2]], %[[C1]]
//   CHECK-DAG:   %[[N3:.+]] = tensor.dim %[[ARG5]], %[[C1]]
//       CHECK:   %[[R0:[a-zA-Z0-9]+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
//  CHECK-SAME:       iter_args(%[[ARG8:[a-zA-Z0-9]+]] = %[[ARG6]], %[[ARG9:[a-zA-Z0-9]+]] = %[[ARG4]], %[[ARG10:[a-zA-Z0-9]+]] = %[[ARG2]])
//   CHECK-DAG:     %[[TILE_M:.+]] = affine.min #[[MAP]](%[[IV]])[%{{.+}}, %[[M]]]
//   CHECK-DAG:     %[[SLICE_ARG0:.+]] = tensor.extract_slice %[[ARG0]][%[[IV]], 0] [%[[TILE_M]], %[[N0]]]
//   CHECK-DAG:     %[[SLICE_ARG1:.+]] = tensor.extract_slice %[[ARG1]][0, 0] [%[[N0]], %[[N1]]]
//   CHECK-DAG:     %[[SLICE_ARG2:.+]] = tensor.extract_slice %[[ARG10]][%[[IV]], 0] [%[[TILE_M]], %[[N1]]]
//   CHECK-DAG:     %[[TILE_GEMM1:.+]] = linalg.matmul ins(%[[SLICE_ARG0]], %[[SLICE_ARG1]] :
//  CHECK-SAME:         outs(%[[SLICE_ARG2]] :
//   CHECK-DAG:     %[[SLICE_ARG3:.+]] = tensor.extract_slice %[[ARG3]][0, 0] [%[[N1]], %[[N2]]]
//   CHECK-DAG:     %[[SLICE_ARG4:.+]] = tensor.extract_slice %[[ARG9]][%[[IV]], 0] [%[[TILE_M]], %[[N2]]]
//   CHECK-DAG:     %[[TILE_GEMM2:.+]] = linalg.matmul ins(%[[TILE_GEMM1]], %[[SLICE_ARG3]] :
//  CHECK-SAME:         outs(%[[SLICE_ARG4]] :
//   CHECK-DAG:     %[[SLICE_ARG5:.+]] = tensor.extract_slice %[[ARG5]][0, 0] [%[[N2]], %[[N3]]]
//   CHECK-DAG:     %[[SLICE_ARG6:.+]] = tensor.extract_slice %[[ARG8]][%[[IV]], 0] [%[[TILE_M]], %[[N3]]]
//   CHECK-DAG:     %[[TILE_GEMM3:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[TILE_GEMM2]], %[[SLICE_ARG5]] :
//  CHECK-SAME:         outs(%[[SLICE_ARG6]] :
//   CHECK-DAG:     %[[UPDATE0:.+]] = tensor.insert_slice %[[TILE_GEMM3]] into %[[ARG8]][%[[IV]], 0] [%[[TILE_M]], %[[N3]]]
//   CHECK-DAG:     %[[UPDATE1:.+]] = tensor.insert_slice %[[TILE_GEMM2]] into %[[ARG9]][%[[IV]], 0] [%[[TILE_M]], %[[N3]]]
//   CHECK-DAG:     %[[UPDATE2:.+]] = tensor.insert_slice %[[TILE_GEMM1]] into %[[ARG10]][%[[IV]], 0] [%[[TILE_M]], %[[N3]]]
//       CHECK:     scf.yield %[[UPDATE0]], %[[UPDATE1]], %[[UPDATE2]]
//       CHECK:   return %[[R0]]#2, %[[R0]]#1, %[[R0]]#0

// -----

func.func @reduction_sequence(%arg0: tensor<30x3xf32>) -> tensor<30x3xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = linalg.init_tensor [30] : tensor<30xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<30xf32>) -> tensor<30xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<30x3xf32>) outs(%1 : tensor<30xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.maxf %arg2, %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<30xf32>
  %3 = linalg.init_tensor [30, 3] : tensor<30x3xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<30xf32>) -> tensor<30xf32>
  %5:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %2 : tensor<30x3xf32>, tensor<30xf32>) outs(%3, %4 : tensor<30x3xf32>, tensor<30xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %8 = arith.subf %arg1, %arg2 : f32
      %9 = math.exp %8 : f32
      %10 = arith.addf %arg4, %9 : f32
      linalg.yield %9, %10 : f32, f32
    } -> (tensor<30x3xf32>, tensor<30xf32>)
  %6 = linalg.generic {
      __internal_linalg_transform__ = "reduction_sequence_fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%5#0, %5#1 : tensor<30x3xf32>, tensor<30xf32>) outs(%3 : tensor<30x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.divf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<30x3xf32>
  return %6 : tensor<30x3xf32>
}
//       CHECK: func @reduction_sequence(%[[ARG0:.+]]: tensor<30x3xf32>)
//   CHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [30]
//   CHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [30, 3]
//       CHECK:   %[[RESULT:[a-zA-Z0-9]+]]:6 = scf.for %[[IV:[a-zA-Z0-9]+]]
//  CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG2:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG3:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG4:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG5:[a-zA-Z0-9]+]] = %[[INIT0]]
//   CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV]], 0]
//   CHECK-DAG:     %[[ITERARG4_SLICE:.+]] = tensor.extract_slice %[[ITERARG4]][%[[IV]]]
//       CHECK:     %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[ITERARG4_SLICE]] :
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]] :
//  CHECK-SAME:         outs(%[[FILL0]] :
//   CHECK-DAG:     %[[ITERARG1_SLICE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV]], 0]
//   CHECK-DAG:     %[[ITERARG5_SLICE:.+]] = tensor.extract_slice %[[ITERARG5]][%[[IV]]]
//       CHECK:     %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[ITERARG5_SLICE]] :
//       CHECK:     %[[GENERIC1:.+]]:2 = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[GENERIC0]] :
//  CHECK-SAME:         outs(%[[ITERARG1_SLICE]], %[[FILL1]] :
//       CHECK:     %[[ITERARG0_SLICE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0]
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]]#0, %[[GENERIC1]]#1 :
//  CHECK-SAME:         outs(%[[ITERARG0_SLICE]] :
//   CHECK-DAG:     %[[INSERTSLICE0:.+]] = tensor.insert_slice %[[GENERIC2]] into %[[ITERARG0]][%[[IV]], 0]
//   CHECK-DAG:     %[[INSERTSLICE1:.+]] = tensor.insert_slice %[[GENERIC1]]#0 into %[[ITERARG1]][%[[IV]], 0]
//   CHECK-DAG:     %[[INSERTSLICE2:.+]] = tensor.insert_slice %[[GENERIC1]]#1 into %[[ITERARG2]][%[[IV]]]
//   CHECK-DAG:     %[[INSERTSLICE3:.+]] = tensor.insert_slice %[[GENERIC0]] into %[[ITERARG3]][%[[IV]]]
//   CHECK-DAG:     %[[INSERTSLICE4:.+]] = tensor.insert_slice %[[FILL0]] into %[[ITERARG4]][%[[IV]]]
//   CHECK-DAG:     %[[INSERTSLICE5:.+]] = tensor.insert_slice %[[FILL1]] into %[[ITERARG5]][%[[IV]]]
//       CHECK:     scf.yield %[[INSERTSLICE0]], %[[INSERTSLICE1]], %[[INSERTSLICE2]], %[[INSERTSLICE3]], %[[INSERTSLICE4]], %[[INSERTSLICE5]]
//       CHECK:   return %[[RESULT]]#0
