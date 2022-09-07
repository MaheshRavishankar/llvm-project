//===- TileUsingInterface.h - Tiling ops using TilingInterface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
#define MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <deque>

namespace mlir {
class Operation;
class PatternRewriter;
class TilingInterface;
} // namespace mlir

namespace mlir {
namespace scf {

using SCFTileSizeComputationFunction =
    std::function<SmallVector<Value>(OpBuilder &, Operation *)>;

/// Options to use to control tiling.
struct SCFTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  SCFTileSizeComputationFunction tileSizeComputationFunction = nullptr;

  SCFTilingOptions &
  setTileSizeComputationFunction(SCFTileSizeComputationFunction fun) {
    tileSizeComputationFunction = std::move(fun);
    return *this;
  }
  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  SCFTilingOptions &setTileSizes(const SmallVector<Value, 4> &ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  SCFTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

  /// The interchange vector to reorder the tiled loops.
  SmallVector<unsigned> interchangeVector = {};
  SCFTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector = llvm::to_vector(interchange);
    return *this;
  }
};

struct SCFTilingResult {
  Operation *tiledOp;
  SmallVector<scf::ForOp> loops;
  llvm::SmallBitVector tiledLoops;
  SmallVector<OpFoldResult> tileOffsets;
  SmallVector<OpFoldResult> tileSizes;
};

/// Pattern to tile an op that implements the `TilingInterface` using
/// `scf.for` for iterating over the tiles.
struct TileUsingSCFForOp : public OpInterfaceRewritePattern<TilingInterface> {
  /// Construct a generic pattern applied to all TilingInterface ops.
  TileUsingSCFForOp(MLIRContext *context, SCFTilingOptions options,
                    PatternBenefit benefit = 1);

  /// Construct a generic pattern applied to `opName`.
  TileUsingSCFForOp(StringRef opName, MLIRContext *context,
                    SCFTilingOptions options, PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<SCFTilingResult>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// Options to control tiling;
  SCFTilingOptions options;
};

/// Pattern to tile and fuse a sequence of operations, by tiling the consumer
/// and fusing its producers. Note that this assumes that it is valid to
/// tile+fuse the producer into the innermost tiled loop. Its up to the caller
/// to ensure that the tile sizes provided make this fusion valid.
///
/// For example, for the following sequence
///
/// ```mlir
/// %0 = linalg.fill ...
/// %1 = linalg.matmul ... outs(%0 : ...) ...
/// ```
///
/// it is legal to fuse the fill with the matmul only if the matmul is tiled
/// along the parallel dimensions and not the reduction dimension, i.e. the tile
/// size for the reduction dimension should be 0.

struct SCFTileAndFuseOptions {
  SCFTilingOptions tilingOptions;
  SCFTileAndFuseOptions &setTilingOptions(SCFTilingOptions options) {
    tilingOptions = options;
    return *this;
  }

  /// When the access pattern between producer and consumer is such that
  /// the producer is fused with the consumer without the producer being
  /// recomputed, then it is possible to yield the value of the producer from
  /// within the loop nest. For example, consider
  //
  /// ```mlir
  /// %0 = linalg.matmul ins(%lhs0, %rhs0) outs(%init0)
  /// %1 = linalg.matmul ins(%0, %rhs1) outs(%init1)
  /// ```
  ///
  /// If the tile sizes chosen are such that the second `linalg.matmul`
  /// is tiled along the outer two dimensions of the op, then fusing
  /// the first `linalg.matmul` using tile and fuse results in
  /// recomputation of parts of the fused producer during computation
  /// of different tiles of the consumer. Instead if only the outer dimension
  /// is chosen, then the producer is not recomputed.
  ///
  /// ```mlir
  /// scf.for %iv0 =
  ///    %lhs0_slice = tensor.extract_slice %lhs0[%iv0, 0]
  ///    %rhs0_slice = tensor.extract_slice %rhs0[0, 0]
  ///    %init0_slice = tensor.extract_slice %init0[%iv0, 0]
  ///    %0 = linalg.matmul ins(%lhs0_slice, %rhs0_slice) outs(%init0_slice)
  ///    %rhs1_slice = tensor.extract_slice %rhs1[0, 0]
  ///    %init1_slice = tensor.extract_slice %init1[%iv0, 0]
  ///    %1 = linalg.matmul ins(%0, %rhs1_slice) outs(%init1_slice)
  /// ```
  ///
  /// If needed the value of the untiled first matmul can be reconstructed
  /// using the tiled and fused operation (similar to how the replacement of the
  /// consumer is done).
  /// It is unclear how to automatically determine when the producer is
  /// recomputed and when it is not (especially through the `TilingInterface`).
  /// So for now this is left as a option for the caller which is expected to
  /// set the tile sizes appropriately to ensure the producer is not recomputed.
  /// With this based on the uses of the untiled producer, a replacement value
  /// for this is yielded by the tiled loop nest.
  ///
  /// One way to ensure this in the producer is when the producer and consumer
  /// are LinalgOps, is
  /// ```cpp
  /// bool canProducerBeFusedWithoutRedundantComputation(
  ///     llvm::SmallBitVector tiledLoops, OpOperand *fusedOperand) {
  ///   auto consumerOp = cast<LinalgOp>(fusedOperand->getOwner());
  ///   AffineMap consumerIndexingMap =
  ///     consumerOp.getTiedIndexingMap(fusedOperand);
  ///   AffineMap projectedConsumerMap =
  ///     getProjectedMap(consumerOp, tiledLoops);
  ///   AffineMap projectedProducerMap =
  ///     getProjectedMap(producerOp, tiledLoops);
  ///   return projectedConsumerMap.isIdentity();
  /// }
  /// ```
  bool producerCanBeFusedWithoutRedundantComputations = false;
  SCFTileAndFuseOptions &
  setProducerCanBeFusedWithoutRedundantComputations(bool val) {
    producerCanBeFusedWithoutRedundantComputations = val;
    return *this;
  }
};
struct SCFTileAndFuseResult {
  SmallVector<Operation *> fusableProducers;
  SmallVector<Operation *> tiledAndFusedOps;
  SmallVector<scf::ForOp> loops;
};

struct TileConsumerAndFuseProducersGreedilyUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {

  /// Construct a generic pattern applied to all TilingInterface ops.
  TileConsumerAndFuseProducersGreedilyUsingSCFForOp(
      MLIRContext *context, SCFTileAndFuseOptions options,
      PatternBenefit benefit = 1);

  /// Construct a generic pattern applied to `opName`.
  TileConsumerAndFuseProducersGreedilyUsingSCFForOp(
      StringRef opName, MLIRContext *context, SCFTileAndFuseOptions options,
      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<SCFTileAndFuseResult>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  SCFTileAndFuseOptions tileAndFuseOptions;
};

/// Pattern to lower operations that implement the `TilingInterface` to
/// loops/scalar IR using `scf.for`.
struct LowerToLoopsUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern<TilingInterface>::OpInterfaceRewritePattern;

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<SmallVector<scf::ForOp>>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }
};

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
