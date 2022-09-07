//===- Tiling.cpp - Implementation of tiling using TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the tiling using TilingInterface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tile-using-interface"

using namespace mlir;

scf::SCFTilingOptions &
scf::SCFTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<func::FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<unsigned>
fillInterchangeVector(ArrayRef<unsigned> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<unsigned> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<unsigned>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Helper method to apply permutation to a vector
template <typename T>
static SmallVector<T> applyPermutationToVector(const SmallVector<T> &vector,
                                               ArrayRef<unsigned> interchange) {
  assert(interchange.size() == vector.size());
  return llvm::to_vector(
      llvm::map_range(interchange, [&](unsigned val) { return vector[val]; }));
}
/// Helper method to apply to invert a permutation.
static SmallVector<unsigned>
invertPermutationVector(ArrayRef<unsigned> interchange) {
  SmallVector<unsigned> inversion(interchange.size());
  for (const auto &pos : llvm::enumerate(interchange)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}
/// Method to check if an interchange vector is a permutation.
static bool isPermutation(ArrayRef<unsigned> interchange) {
  llvm::SmallDenseSet<unsigned, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val))
      return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

//===----------------------------------------------------------------------===//
// TileUsingSCFForOp pattern implementation.
//===----------------------------------------------------------------------===//

// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  Optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  Optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  Optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the
///   tile processed within the inner most loop.
static SmallVector<scf::ForOp>
generateTileLoopNest(OpBuilder &builder, Location loc,
                     ArrayRef<Range> loopRanges, ArrayRef<Value> tileSizeVals,
                     SmallVector<OpFoldResult> &offsets,
                     SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp> loops;
  offsets.resize(loopRanges.size());
  sizes.resize(loopRanges.size());

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable
  // of the tiled loop.
  AffineExpr s0, s1, d0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());

  for (auto loopRange : llvm::enumerate(loopRanges)) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) {
      offsets[loopRange.index()] = offset;
      sizes[loopRange.index()] = size;
      continue;
    }

    auto loop = builder.create<scf::ForOp>(
        loc, offset, size, tileSizeVals[loopRange.index()], ValueRange{},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          bool canAvoidMap = tileDividesIterationDomain(
              Range{loopRange.value().offset, loopRange.value().size,
                    tileSizeVals[loopRange.index()]});
          Value boundedTileSize =
              (canAvoidMap)
                  ? tileSizeVals[loopRange.index()]
                  : builder.create<AffineMinOp>(
                        bodyLoc, minMap,
                        ValueRange{iv, tileSizeVals[loopRange.index()], size});
          sizes[loopRange.index()] = boundedTileSize;
          builder.create<scf::YieldOp>(loc);
        });
    offsets[loopRange.index()] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return loops;
}

/// If producers of an destination operand are fused with its consumer
/// then while yielding the value of the consumer, the destination of
/// the producer is preferable to use as the initial value for the
/// iter_arg. This removes an unnecessary use of the producer as the
/// init value of the yielded result. For example, for this sequence
///
/// ```mlir
///   %0 = linalg.init_tensor ...
///   %1 = linalg.fill ins(...) outs(%0 : ...)
///   %2 = linalg.matmul ins(...) outs(%1 : ...)
/// ```
///
/// when tiled and fused
///
/// ```mlir
/// %0 = linalg.init_tensor
/// %1 = scf.for ... iter_args(%arg0 = %0)
///  %2 = scf.for ... iter_args(%arg1 = %arg0)
///     %3 = tensor.extract_slice %arg1 ...
///     %4 = linalg.fill ins(...) outs(%3 : ...)
///     %5 = linalg.matmul ins(...) outs(%4 : ...)
/// ```
///
/// is preferable to
///
/// ```mlir
/// %0 = linalg.init_tensor
/// %1 = linalg.fill ... outs(%0)
/// %2 = scf.for ... iter_args(%arg0 = %1)
///  %3 = scf.for ... iter_args(%arg1 = %arg0)
///     %3 = tensor.extract_slice %arg1 ...
///     %4 = linalg.fill ins(...) outs(%3 : ...)
///     %5 = linalg.matmul ins(...) outs(%4 : ...)
/// ```
///
/// This method tracks through the tiled and fused ops to get to the
/// actual value to use for the destination.
// TODO: This will be much easier to do when the DestinationPassingStyle
// interface is moved to `mlir/Interfaces`.
static Value getTileDestination(OpBuilder &builder, OpResult result) {
  Operation *definingOp = result.getOwner();
  while (definingOp) {
    Value destinationTile =
        TypeSwitch<Operation *, Value>(definingOp)
            .Case<TilingInterface>([&](auto interfaceOp) {
              return interfaceOp.getDestinationOperands(
                  builder)[result.getResultNumber()];
            })
            .Default([&](Operation *) -> Value { return nullptr; });
    definingOp = destinationTile ? destinationTile.getDefiningOp() : nullptr;
    result = definingOp ? destinationTile.cast<OpResult>() : result;
  }
  return result;
}

/// Given a list of `untiledOps` that are the original untiled operations, and
/// their tiled counterparts (`tiledOps`) yield the values of the results
/// through the generated `tilingLoops`. The result of the outermost loop forms
/// a replacement for the `untiledOps`. This method performs the replacement as
/// well.
static LogicalResult yieldTiledValues(RewriterBase &rewriter,
                                      ArrayRef<TilingInterface> untiledOps,
                                      ArrayRef<Operation *> tiledOps,
                                      ArrayRef<OpFoldResult> tileOffsets,
                                      ArrayRef<OpFoldResult> tileSizes,
                                      MutableArrayRef<scf::ForOp> tilingLoops) {
  SmallVector<tensor::ExtractSliceOp> destSliceOps;
  SmallVector<Value> initValues;
  for (auto tiledOp : llvm::enumerate(tiledOps)) {
    for (auto result : llvm::enumerate(tiledOp.value()->getResults())) {
      Value destinationTile = getTileDestination(rewriter, result.value());
      tensor::ExtractSliceOp sliceOp =
          destinationTile
              ? destinationTile.getDefiningOp<tensor::ExtractSliceOp>()
              : nullptr;
      destSliceOps.push_back(sliceOp);
      if (sliceOp) {
        initValues.push_back(sliceOp.getSource());
      } else {
        TilingInterface untiledOp = untiledOps[tiledOp.index()];
        initValues.push_back(
            untiledOp.getDestinationOperands(rewriter)[result.index()]);
      }
    }
  }
  // 5b. `scf.for` with tensor semantics requires the loop nest to yield the
  // replacement values using destructive updates. Use the `TilingInterface`
  // to get the position of the result tiles and use that to generate the
  // destructive update pattern, i.e.,
  //
  // ```mlir
  // scf.for %iv0 = ... {
  //   %0 = tiled_op
  // }
  // ```
  //
  // is transformed to
  //
  // ```mlir
  // %result = scf.for %iv0 = ... iter_args(%arg = %init) -> .. {
  //   %0 = tiled_op
  //   %1 = tensor.insert_slice %0 into %arg[..] [..] [..]
  //   scf.yield %1
  // }
  // ```
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> yieldedValues;
    Attribute one = b.getIndexAttr(1);
    Attribute zero = b.getIndexAttr(0);
    unsigned bbArgNum = 0;
    for (auto it : llvm::enumerate(untiledOps)) {
      TilingInterface untiledOp = it.value();
      SmallVector<OpFoldResult> opTileOffsets(tileOffsets),
          opTileSizes(tileSizes);
      unsigned opRank = untiledOp.getLoopIteratorTypes().size();
      opTileOffsets.resize(opRank, zero);
      opTileSizes.resize(opRank, zero);
      for (auto source : llvm::enumerate(tiledOps[it.index()]->getResults())) {
        SmallVector<OpFoldResult> resultTileOffsets, resultTileSizes;
        if (failed(untiledOp.getResultTilePosition(
                b, source.index(), opTileOffsets, opTileSizes,
                resultTileOffsets, resultTileSizes))) {
          return {};
        }

        SmallVector<OpFoldResult> resultTileStrides(resultTileOffsets.size(),
                                                    one);
        tensor::ExtractSliceOp destSliceOp = destSliceOps[bbArgNum];
        if (destSliceOp && destSliceOp->getBlock() == b.getInsertionBlock())
          destSliceOp.setOperand(0, newBBArgs[bbArgNum]);
        Value yieldedValue = b.create<tensor::InsertSliceOp>(
            loc, source.value(), newBBArgs[bbArgNum++], resultTileOffsets,
            resultTileSizes, resultTileStrides);
        yieldedValues.push_back(yieldedValue);
      }
    }
    return yieldedValues;
  };

  // Modify the loop nest to yield the result values.
  SmallVector<scf::ForOp> newLoops = replaceLoopNestWithNewYields(
      rewriter, tilingLoops, initValues, yieldValueFn);
  for (const auto &loop : llvm::enumerate(tilingLoops)) {
    rewriter.eraseOp(loop.value());
    tilingLoops[loop.index()] = newLoops[loop.index()];
  }
  scf::ForOp outerMost = tilingLoops.front();
  llvm::SmallDenseSet<Operation *> untiledOpsSet;
  untiledOpsSet.insert(untiledOps.begin(), untiledOps.end());
  unsigned resultNum = 0;
  for (auto untiledOp : untiledOps) {
    if (untiledOp->getNumResults() + resultNum > outerMost.getNumResults()) {
      return rewriter.notifyMatchFailure(
          untiledOp, "failed to yield results of operation");
    }
    // Replace all uses of the untiled op with values returned from the
    // loop except for in `tensor.dim` operations or other ops that are fused
    // here. Those need to be resolved separately.
    // TODO: Find a better way to handle replacements.
    rewriter.replaceOpWithIf(
        untiledOp,
        outerMost.getResults().drop_front(resultNum).take_front(
            untiledOp->getNumResults()),
        [&](OpOperand &use) {
          Operation *user = use.getOwner();
          return !untiledOpsSet.count(user) &&
                 !isa<tensor::DimOp>(use.getOwner());
        });
    resultNum += untiledOp->getNumResults();
  }
  return success();
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
static FailureOr<scf::SCFTilingResult>
tileConsumer(RewriterBase &rewriter, TilingInterface op,
             scf::SCFTilingOptions options) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<Value> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  scf::SCFTilingResult tilingResult;
  tilingResult.tiledLoops.resize(numLoops);
  for (auto tileSize : llvm::enumerate(tileSizeVector))
    if (!isConstantIntValue(tileSize.value(), 0))
      tilingResult.tiledLoops.set(tileSize.index());

  SmallVector<OpFoldResult> offsets, sizes;
  {
    // If there is an interchange specified, permute the iteration domain and
    // the tile sizes.
    SmallVector<unsigned> interchangeVector;
    if (!options.interchangeVector.empty()) {
      interchangeVector = fillInterchangeVector(options.interchangeVector,
                                                iterationDomain.size());
    }
    if (!interchangeVector.empty()) {
      if (!isPermutation(interchangeVector)) {
        return rewriter.notifyMatchFailure(
            op, "invalid intechange vector, not a permutation of the entire "
                "iteration space");
      }

      iterationDomain =
          applyPermutationToVector(iterationDomain, interchangeVector);
      tileSizeVector =
          applyPermutationToVector(tileSizeVector, interchangeVector);
    }

    // 3. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    tilingResult.loops = generateTileLoopNest(
        rewriter, op.getLoc(), iterationDomain, tileSizeVector, offsets, sizes);

    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      offsets = applyPermutationToVector(offsets, inversePermutation);
      sizes = applyPermutationToVector(sizes, inversePermutation);
    }

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "LoopNest shell :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });

    // 4. Generate the tiled implementation within the inner most loop.
    if (!tilingResult.loops.empty())
      rewriter.setInsertionPoint(
          tilingResult.loops.back().getBody()->getTerminator());
    SmallVector<Operation *> tiledImplementation =
        op.getTiledImplementation(rewriter, offsets, sizes);
    if (tiledImplementation.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected tiled implementation to return a single op");
    }
    tilingResult.tiledOp = tiledImplementation[0];
    std::swap(tilingResult.tileOffsets, offsets);
    std::swap(tilingResult.tileSizes, sizes);

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "After tiled implementation :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });
  }

  return tilingResult;
}

scf::TileUsingSCFForOp::TileUsingSCFForOp(MLIRContext *context,
                                          scf::SCFTilingOptions options,
                                          PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      options(std::move(options)) {}

scf::TileUsingSCFForOp::TileUsingSCFForOp(StringRef opName,
                                          MLIRContext *context,
                                          scf::SCFTilingOptions options,
                                          PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      options(std::move(options)) {}

FailureOr<scf::SCFTilingResult>
scf::TileUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {

  FailureOr<scf::SCFTilingResult> tilingResult =
      tileConsumer(rewriter, op, options);
  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(op, "failed to tile operation");

  // If there are no results (i.e. buffer semantics), there is nothing to do.
  // Erase op and return.
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
    return tilingResult;
  }

  // If there are no loops, there is nothing more to do.
  if (tilingResult->loops.empty()) {
    // Replace the original op with the tiled op.
    rewriter.replaceOp(op, tilingResult->tiledOp->getResults());
    return tilingResult;
  }

  // From the tiled ops reconstruct the value to replace the result of the
  // untiled op using destructive updates.
  if (failed(yieldTiledValues(rewriter, op, tilingResult->tiledOp,
                              tilingResult->tileOffsets,
                              tilingResult->tileSizes, tilingResult->loops)))
    return failure();
  return tilingResult;
}

//===----------------------------------------------------------------------===//
// TileConsumerAndFuseProducersGreedilyUsingSCFForOp pattern implementation.
//===----------------------------------------------------------------------===//

scf::TileConsumerAndFuseProducersGreedilyUsingSCFForOp::
    TileConsumerAndFuseProducersGreedilyUsingSCFForOp(
        MLIRContext *context, scf::SCFTileAndFuseOptions options,
        PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      tileAndFuseOptions(std::move(options)) {}

scf::TileConsumerAndFuseProducersGreedilyUsingSCFForOp::
    TileConsumerAndFuseProducersGreedilyUsingSCFForOp(
        StringRef opName, MLIRContext *context,
        scf::SCFTileAndFuseOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      tileAndFuseOptions(std::move(options)) {}

/// Collect all transitive producers of `op` and return then in sorted order
/// (i.e def before use).
static SmallVector<Operation *> collectTransitiveProducers(TilingInterface op) {
  SetVector<Operation *> visited;
  SmallVector<Operation *> worklist;
  SmallVector<Operation *> sortedOps;
  worklist.push_back(op);
  while (!worklist.empty()) {
    TilingInterface currOp = worklist.back();
    bool addedProducer = false;
    for (OpOperand &operand : currOp->getOpOperands()) {
      auto producerOp = operand.get().getDefiningOp<TilingInterface>();
      if (!producerOp || visited.count(producerOp))
        continue;
      addedProducer = true;
      worklist.push_back(producerOp);
      visited.insert(producerOp);
    }
    if (!addedProducer) {
      if (op != currOp)
        sortedOps.push_back(currOp);
      worklist.pop_back();
    }
  }
  return sortedOps;
}

/// For a given untiled op `producer` find all instances where
/// slices of this operation are used in `tiledOps`.
struct CandidateSliceOp {
  tensor::ExtractSliceOp sliceOp;
  OpResult producerResult;
};
static SmallVector<CandidateSliceOp>
collectAllSlicesToProducer(ArrayRef<Operation *> tiledOps,
                           TilingInterface producer) {
  SmallVector<CandidateSliceOp> slicesOfProducer;
  for (auto tiledOp : tiledOps) {
    for (OpOperand &operand : tiledOp->getOpOperands()) {
      auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
      if (!sliceOp)
        continue;

      Value source = sliceOp.getSource();
      while (auto blockArg = source.dyn_cast<BlockArgument>()) {
        auto loopOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (!loopOp)
          break;
        source = loopOp.getOpOperandForRegionIterArg(blockArg).get();
      }

      if (source.getDefiningOp() == producer.getOperation()) {
        slicesOfProducer.emplace_back(
            CandidateSliceOp{sliceOp, source.cast<OpResult>()});
      }
    }
  }
  return slicesOfProducer;
}

/// Implementation of the the steps to fuse an untiled `producer` with
/// all uses of it in `tiledOps`. If `isFusableWithRedundantComputation` is
/// - `false` : each slice is replaced with a tiled version of the producer
///             that produces the
/// - `true`  : it is assumed that a single instance of tiling the producer
///             can be used to replace all (slice) uses of the untiled
///             producer in `tiledOps`.
static FailureOr<SmallVector<Operation *>>
fuseProducer(RewriterBase &rewriter, ArrayRef<Operation *> tiledOps,
             TilingInterface producer,
             bool isFusableWithoutRedundantComputation = false) {
  SmallVector<Operation *> fusedProducers;
  SmallVector<CandidateSliceOp> slicesOfProducer =
      collectAllSlicesToProducer(tiledOps, producer);
  if (slicesOfProducer.empty())
    return fusedProducers;

  if (!isFusableWithoutRedundantComputation) {
    // Simpler usage, with lesser constraints. Just replace each slice
    // with tiled implementation of the producer.
    for (auto candidateSliceOp : slicesOfProducer) {
      tensor::ExtractSliceOp sliceOp = candidateSliceOp.sliceOp;
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(sliceOp);
      FailureOr<Value> fusedProducerValue =
          tensor::replaceExtractSliceWithTiledProducer(
              rewriter, sliceOp, candidateSliceOp.producerResult);
      if (failed(fusedProducerValue)) {
        // The fusion failed. Continue fusing other slices.
        continue;
      }
      rewriter.replaceOp(sliceOp, fusedProducerValue.value());
      fusedProducers.push_back(fusedProducerValue->getDefiningOp());
    }
    return fusedProducers;
  }

  // Assume one instance of the tiled producer can replace all uses in
  // `tiledOps`. Take the first slice op and use that to produce the tiled
  // implementation.
  auto currSliceOp = slicesOfProducer.front().sliceOp;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(currSliceOp);
  FailureOr<Value> fusedProducerVal =
      tensor::replaceExtractSliceWithTiledProducer(
          rewriter, currSliceOp, slicesOfProducer.front().producerResult);
  if (failed(fusedProducerVal)) {
    // The fusion failed. Continue fusing other ops.
    return fusedProducers;
  }
  TilingInterface tiledProducer =
      fusedProducerVal->getDefiningOp<TilingInterface>();
  if (!tiledProducer ||
      tiledProducer->getNumResults() != producer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        producer,
        "unhandled case where tiled implementation does not return a single "
        "operation with as many results as the untiled operation");
  }
  fusedProducers.push_back(tiledProducer);

  // 3c. Replace the slice uses with the corresponding producer use.
  for (auto candidateSliceOp : slicesOfProducer) {
    unsigned resultNumber = candidateSliceOp.producerResult.getResultNumber();
    rewriter.replaceOp(candidateSliceOp.sliceOp,
                       tiledProducer->getResult(resultNumber));
  }
  return fusedProducers;
}

FailureOr<scf::SCFTileAndFuseResult>
scf::TileConsumerAndFuseProducersGreedilyUsingSCFForOp::
    returningMatchAndRewrite(TilingInterface op,
                             PatternRewriter &rewriter) const {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!op->getNumResults()) {
    return rewriter.notifyMatchFailure(
        op, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SCFTileAndFuseResult tileAndFuseResult;
  SmallVector<OpFoldResult> consumerTileOffsets, consumerTileSizes;
  {
    FailureOr<SCFTilingResult> tilingResult =
        tileConsumer(rewriter, op, tileAndFuseOptions.tilingOptions);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(op, "failed to tile consumer");
    tileAndFuseResult.tiledAndFusedOps.push_back(tilingResult->tiledOp);
    tileAndFuseResult.loops = std::move(tilingResult->loops);
    std::swap(tilingResult->tileOffsets, consumerTileOffsets);
    std::swap(tilingResult->tileSizes, consumerTileSizes);
  }

  // If there are no loops generated, fusion is immaterial.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

  // 2. Collect a list of producers of the original operation that are to be
  // tiled and fused.
  tileAndFuseResult.fusableProducers = collectTransitiveProducers(op);

  // 3. Iterate through the producers in reverse and tile and fuse them.
  for (Operation *producerOp :
       llvm::reverse(tileAndFuseResult.fusableProducers)) {
    auto producer = cast<TilingInterface>(producerOp);

    FailureOr<SmallVector<Operation *>> fusedProducers = fuseProducer(
        rewriter, tileAndFuseResult.tiledAndFusedOps, producer,
        tileAndFuseOptions.producerCanBeFusedWithoutRedundantComputations);
    if (failed(fusedProducers)) {
      // Fusion failed. Continue with other producers.
      continue;
    }

    if (tileAndFuseOptions.producerCanBeFusedWithoutRedundantComputations &&
        fusedProducers->size() != 1) {
      return rewriter.notifyMatchFailure(
          producer, "expected single operation for the fused producer");
    }

    tileAndFuseResult.tiledAndFusedOps.append(fusedProducers.value());
  }

  // 4. Finally reconstruct the replacements for the untiled operations
  // using destructive updates. If
  // `producerCanBeFusedWithoutRedundantComputation` is
  // - `true` : Yield the results of all the producers. It is assumed
  //            to be valid.
  // - `false` : Yield the results of just the tiled consumer.
  SmallVector<TilingInterface> untiledOps;
  ArrayRef<Operation *> tiledOps = {tileAndFuseResult.tiledAndFusedOps.front()};
  untiledOps.push_back(op);
  if (tileAndFuseOptions.producerCanBeFusedWithoutRedundantComputations) {
    if (tileAndFuseResult.tiledAndFusedOps.size() !=
        tileAndFuseResult.fusableProducers.size() + 1) {
      return rewriter.notifyMatchFailure(
          op, "expected as many tiled and fused ops as producer");
    }
    untiledOps.append(tileAndFuseResult.fusableProducers.rbegin(),
                      tileAndFuseResult.fusableProducers.rend());
    tiledOps = llvm::makeArrayRef(tileAndFuseResult.tiledAndFusedOps);
  }
  if (failed(yieldTiledValues(rewriter, untiledOps, tiledOps,
                              consumerTileOffsets, consumerTileSizes,
                              tileAndFuseResult.loops))) {
    return rewriter.notifyMatchFailure(op, "failed to yield values");
  }

  return tileAndFuseResult;
}

//===----------------------------------------------------------------------===//
// LowerToLoopsUsingSCFForOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<scf::ForOp>>
scf::LowerToLoopsUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  SmallVector<Range> domain = op.getIterationDomain(rewriter);

  // TODO: Handle cases where the op has results if needed.
  if (op->getNumResults() > 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to lower to loops operations with return values");
  }

  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> loops;
  Location loc = op.getLoc();
  for (auto loopRange : domain) {
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value sizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value strideVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.stride);
    auto loop = rewriter.create<scf::ForOp>(op.getLoc(), offsetVal, sizeVal,
                                            strideVal, ValueRange{});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  if (failed(op.generateScalarImplementation(rewriter, op.getLoc(), ivs))) {
    return failure();
  }
  rewriter.eraseOp(op);
  return loops;
}
