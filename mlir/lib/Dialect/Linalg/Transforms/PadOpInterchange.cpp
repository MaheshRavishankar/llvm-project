//===- PadOpInterchange.cpp - Interchange pad operation with Generic ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pattenrs that intechanges a generic op -> pad_tensor
// pattern into extract_slice -> generic_op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct FusePadTensorOp : OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PadTensorOp padOp,
                                PatternRewriter &rewriter) const override {
    // Only works on padding op that sets the padded value to a constant.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return rewriter.notifyMatchFailure(padOp,
                                         "only supported for constant padding");
    }
    // This pattern could work for any Linalg op. For now restrict it to generic
    // ops.
    Value source = padOp.source();
    auto genericOp = source.getDefiningOp<GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(
          padOp, "expected source to be linalg.generic op");
    }
    // This pattern could work for op with any iterator types. For now restrict
    // to ops with only parallel iterator type.
    if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
      return rewriter.notifyMatchFailure(
          padOp, "only supported for ops with all parallel iterator types");
    }
    ReifiedRankedShapedTypeDims resultShape;
    if (failed(padOp.reifyResultShapes(rewriter, resultShape)) ||
        resultShape.size() != 1) {
      return rewriter.notifyMatchFailure(
          padOp, "failed to get shape of pad op result");
    }

    Location loc = rewriter.getFusedLoc({genericOp.getLoc(), padOp.getLoc()});

    // Create the tensor of same size as output of the pad op.
    RankedTensorType padResultType = padOp.getResultType();
    auto resultSizes = getAsOpFoldResult(resultShape[0]);
    auto initTensor = rewriter.create<InitTensorOp>(
        loc, resultSizes, padResultType.getElementType());

    // Fill the tensor with the pad value.
    auto fillTensor =
        rewriter.create<FillOp>(loc, padValue, initTensor.getResult());

    // Construct a slice of the fill result that is to be replaced with the
    // result of the generic op. The low pad values are the offsets, the size of
    // the source is the size of the slice.
    unsigned resultNumber = source.cast<OpResult>().getResultNumber();
    SmallVector<OpFoldResult> offsets = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(offsets.size());
    for (auto shape : llvm::enumerate(
             source.getType().cast<RankedTensorType>().getShape())) {
      if (ShapedType::isDynamic(shape.value())) {
        sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, source, shape.index())
                .getResult());
        continue;
      }
      sizes.push_back(rewriter.getIndexAttr(shape.value()));
    }
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, fillTensor.getResult(0), offsets, sizes, strides);

    // Clone the generic op.
    auto clonedOp = cast<GenericOp>(rewriter.clone(*genericOp.getOperation()));
    clonedOp.setOutputOperand(resultNumber, slice.getResult());

    auto insertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, clonedOp.getResult(resultNumber), fillTensor.getResult(0), offsets,
        sizes, strides);
    rewriter.replaceOp(padOp, insertOp.getResult());
    return success();
  }
};
} // namespace

void mlir::linalg::populateFusePadTensorWithGenericOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FusePadTensorOp>(patterns.getContext());
}
