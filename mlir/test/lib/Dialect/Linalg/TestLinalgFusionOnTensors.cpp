//===- TestLinalgFusionOnTensorsPass.cpp - Test Linalg fusion patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg fusion patterns on tensors.
// Specifically the `tileConsumerAndFuseProducers` method.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgFusionOnTensorsPass
    : public PassWrapper<TestLinalgFusionOnTensorsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgFusionOnTensorsPass)

  StringRef getArgument() const final {
    return "test-linalg-tile-and-fuse-on-tensors";
  }

  StringRef getDescription() const final {
    return "Test Linalg tiling and fusion on tensor operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }
  TestLinalgFusionOnTensorsPass() = default;
  TestLinalgFusionOnTensorsPass(const TestLinalgFusionOnTensorsPass &pass) {}

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();

    RewritePatternSet fusionPatterns(context);

    fusionPatterns.insert<LinalgTileAndFuseTensorOpsPattern>(
        context,
        LinalgTilingAndFusionOptions()
            .setTileSizes({10, 20})
            .setReturnFusedOpValues(true),
        LinalgTransformationFilter(
            StringAttr::get(context, "return_fused_values"),
            StringAttr::get(context, "fused_ops")));

    fusionPatterns.insert<LinalgTileAndFuseTensorOpsPattern>(
        context,
        LinalgTilingAndFusionOptions()
            .setTileSizes({10, 0, 0})
            .setReturnFusedOpValues(true),
        LinalgTransformationFilter(StringAttr::get(context, "lhs_fusion"),
                                   StringAttr::get(context, "fused_ops")));

    fusionPatterns.insert<LinalgTileAndFuseTensorOpsPattern>(
        context,
        LinalgTilingAndFusionOptions()
            .setTileSizes({0, 20, 0})
            .setReturnFusedOpValues(true),
        LinalgTransformationFilter(StringAttr::get(context, "rhs_fusion"),
                                   StringAttr::get(context, "fused_ops")));

    fusionPatterns.insert<LinalgTileAndFuseTensorOpsPattern>(
        context,
        LinalgTilingAndFusionOptions()
            .setTileSizes({10, 20, 0})
            .setReturnFusedOpValues(true),
        LinalgTransformationFilter(
            StringAttr::get(context, "matmul_outs_fusion"),
            StringAttr::get(context, "fused_ops")));

    (void)applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgFusionOnTensorsPass() {
  PassRegistration<TestLinalgFusionOnTensorsPass>();
}
} // namespace test
} // namespace mlir
