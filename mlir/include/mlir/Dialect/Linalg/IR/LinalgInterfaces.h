//===- LinalgInterface.h - Linalg operations interfaces -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for Linalg operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
#define MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

/// Get the loop dimension that iterates over the batch dimension for `op` that
/// implements the `ConvolutionOpInterface`. Returns llvm::None if the `op` does
/// not implement the `ConvolutionOpInterface`, or if no batch dimensions exist.
Optional<unsigned> getConvolutionBatchLoopDim(Operation *op);

/// Get the loop dimensions that iterate over the output image for `op` that
/// implements the `ConvolutionOpInterface`. Returns `{}` if the `op` does not
/// implement the `ConvolutionOpInterface`.
SmallVector<unsigned> getConvolutionOutputImageLoopDims(Operation *op);

/// Get the loop dimension that iterates over the output channel dimensions for
/// `op` that implements the `ConvolutionOpInterface`.  Returns llvm::None if
/// the `op` does not implement the `ConvolutionOpInterface`, or if no output
/// channel dimensions exist.
Optional<unsigned> getConvolutionOutputChannelLoopDim(Operation *op);

/// Get the loop dimensions that iterate over the filter loops for `op` that
/// implements the `ConvolutionOpInterface`. Returns `{}` if the `op` does not
/// implement the `ConvolutionOpInterface`.
SmallVector<unsigned> getConvolutionFilterLoopDims(Operation *op);

/// Get the loop dimension that iterates over the input channel dimensions for
/// `op` that implements the `ConvolutionOpInterface`.  Returns llvm::None if
/// the `op` does not implement the `ConvolutionOpInterface`, or if no input
/// channel dimensions exist.
Optional<unsigned> getConvolutionInputChannelLoopDim(Operation *op);

/// Get the loop dimension that iterates over the depthwise dimension for `op`
/// that implements the `ConvolutionOpInterface`.  Returns llvm::None if the
/// `op` does not implement the `ConvolutionOpInterface`, or is not a depthwise
/// convolution.
Optional<unsigned> getConvolutionDepthwiseLoopDim(Operation *op);

namespace linalg {
class LinalgOp;

/// OpOperand vector that implicitly converts to a Value vector.
struct OpOperandVector : public SmallVector<OpOperand *> {
  operator SmallVector<Value>();
};

/// Returns the values obtained by applying `map` to the list of values.
SmallVector<Value, 4> applyMapToValues(OpBuilder &b, Location loc,
                                       AffineMap map, ValueRange values);

/// Checks whether `linalgOp` conforms to ContractionOpInterface.
// TODO: embed within `isa<ContractionOpInterface>` if possible / natural.
bool isaContractionOpInterface(LinalgOp linalgOp);

namespace detail {

/// Verify that `op` conforms to ContractionOpInterface.
LogicalResult verifyContractionInterface(Operation *op);

/// Verify that `op` conforms to the ConvolutionOpInterface.
LogicalResult verifyConvolutionInterface(Operation *op);

/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyStructuredOpInterface(Operation *op);

} // namespace detail
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

/// Include the generated interface declarations.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
