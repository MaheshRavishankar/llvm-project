//===- FusionOnTensors.cpp - Implementation of linalg Fusion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg fusion on tensors
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace linalg;

//===----------------------------------------------------------------------===//
// StructuredOp specific helpers.
//===----------------------------------------------------------------------===//

/// Returns the tiled slice dimensions given the tiled consumer loop dimensions.
/// The slice defines a hyper rectangular iteration space and fusing the
/// producer is always possible. However, depending on the consumer indexing
/// map, not all slice elements may be consumed and the tiles may overlap. In
/// these cases, fusion introduces redundant computation.
static SmallVector<int64_t> getTiledSliceDims(OpOperand *consumerOperand,
                                              ArrayRef<int64_t> tiledLoopDims) {
  // Get the consumer operand indexing map.
  LinalgOp consumerOp = consumerOperand->getOwner();
  AffineMap indexingMap = consumerOp.getTiedIndexingMap(consumerOperand);

  // Search the slice dimensions tiled by a tile loop dimension.
  DenseSet<int64_t> tiledSliceDimIndices;
  for (const auto &en : enumerate(indexingMap.getResults())) {
    for (auto tiledLoopDim : tiledLoopDims) {
      if (en.value().isFunctionOfDim(tiledLoopDim))
        tiledSliceDimIndices.insert(en.index());
    }
  }
  return {tiledSliceDimIndices.begin(), tiledSliceDimIndices.end()};
}

/// Given a vector of `tiledSliceDimIndices` that represent the tiled dimensions
/// of the producer result slice returns the tiled producer loop dimensions.
/// Example:
/// ```
/// %res = linalg.fill(%cst, %input)
/// scf.for %i
///   scf.for %j
///     %slice = tensor.extract_slice %res[%i, %j]
/// ```
/// getTiledProducerLoops(%res, [0, 1]) returns the loop indices [0, 1].
static SmallVector<int64_t>
getTiledProducerLoops(OpResult producerResult,
                      ArrayRef<int64_t> tiledSliceDimIndices) {
  LinalgOp producerOp = producerResult.getOwner();

  // Get the indexing map of the `producerOp` output operand that matches
  // ´producerResult´.
  AffineMap producerIndexingMap = producerOp.getTiedIndexingMap(
      producerOp.getOutputOperand(producerResult.getResultNumber()));

  // Keep only the tiled result slice dimensions of `producerIndexingMap`.
  AffineMap tiledProducerIndexingSubMap =
      producerIndexingMap.getSubMap(SmallVector<unsigned>(
          tiledSliceDimIndices.begin(), tiledSliceDimIndices.end()));

  // Compute the producer loop indices mapped to the tiled result slice
  // dimensions. As the output indexing map of structured operations are
  // projected permutations, `tiledProducerIndexingSubMap` has to be a
  // projected permutation as well. We can thus obtain the producer loop indices
  // by getting the positions of the result dimensions.
  // Example:
  // (d0, d1, d2) -> (d0, d2) has the result positions [0, 2].
  assert(tiledProducerIndexingSubMap.isProjectedPermutation() &&
         "expect slice and producer loop dimensions map one-to-one");
  SmallVector<int64_t> tiledProducerLoopIndices;
  llvm::transform(
      llvm::seq<unsigned>(0, tiledProducerIndexingSubMap.getNumResults()),
      std::back_inserter(tiledProducerLoopIndices), [&](unsigned idx) {
        return tiledProducerIndexingSubMap.getDimPosition(idx);
      });

  return tiledProducerLoopIndices;
}

/// Returns the producer fused in place of `sliceOp`. Tile the producer operands
/// along the `tiledSliceDimIndices` and clone the producer. Consider the case
/// of fusion of an output tensor:
/// ```
/// %1 = producer ins(...) outs(%0)
/// %2 = consumer ins(...) outs(%1)
/// ```
/// When consumer is tiled, %1 appears in the loop iter_args:
/// ```
/// %1 = producer ins(...) outs(%0)
/// %2 = scf.for ... iter_args(%1) .. (%bbarg) {
///   %t1 = tensor.extract_slice %bbarg[..]
///   %t2 = consumer ins(...) outs(%t1)
///   %r = tensor.insert_slice %t2, %bbarg[...]
/// }
/// ```
/// Fusing %1 into the loop requires updating iter_args(%1) to iter_args(%0):
/// ```
/// %2 = scf.for ... iter_args(%0) .. (%bbarg) {
///   %t0 = tensor.extract_slice %bbarg[..]
///   %t1 = producer ins(...) outs(%t0)
///   %t2 = consumer ins(...) outs(%t1)
///   %r = tensor.insert_slice %t2, %bbarg[...]
/// }
/// ```
/// This transformation is only valid if %bbarg is exclusively used by the
/// output ExtractSliceOp / InsertSliceOp pair, which is checked by the
/// `fuseProducer` method.
/// TODO: instead of check and failure, insert new iter_args each time a
/// producer is fused into a consumer and fold away unused iter_args.
static LinalgOp getTiledProducer(OpBuilder &b, OpResult producerResult,
                                 tensor::ExtractSliceOp sliceOp,
                                 ArrayRef<int64_t> tiledSliceDimIndices,
                                 ArrayRef<int64_t> tiledProducerLoopIndices,
                                 OpOperand *iterArg) {
  // Clone the producer after `sliceOp` since the slice may be reused to pass in
  // the producer result.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(sliceOp);

  // Get the producer.
  LinalgOp producerOp = producerResult.getOwner();
  Location loc = producerOp.getLoc();

  // Obtain the `producerOp` loop bounds and the `sliceOp` ranges.
  SmallVector<Value> producerLoopBounds;
  llvm::transform(producerOp.createLoopRanges(b, loc),
                  std::back_inserter(producerLoopBounds),
                  [](Range range) { return range.size; });
  SmallVector<Range> sliceOpRanges = sliceOp.getOrCreateRanges(b, loc);

  // Tile the producer operands given the `sliceOp` ranges. Iterate the
  // `tiledSliceDimIndices` and store the tile offset and size for the tiled
  // slice dimension.
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileIvs(producerOp.getNumLoops(), nullptr);
  SmallVector<Value> tileSizes(producerOp.getNumLoops(), zero);
  SmallVector<Value> allIvs(producerOp.getNumLoops(), nullptr);
  for (auto it : zip(tiledSliceDimIndices, tiledProducerLoopIndices)) {
    int64_t tiledSliceDim = std::get<0>(it);
    int64_t tiledProducerLoop = std::get<1>(it);
    tileIvs[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].offset;
    tileSizes[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].size;
    allIvs[tiledProducerLoop] = tileIvs[tiledProducerLoop];
  }
  erase_value(tileIvs, nullptr);
  SmallVector<Value> tiledOperands = producerOp.getInputAndOutputOperands();
  tiledOperands = makeTiledShapes(b, loc, producerOp, tiledOperands, tileIvs,
                                  tileSizes, producerLoopBounds,
                                  /**omitPartialTileCheck=*/false);

  // Output fusion has to update the iteration arguments of the tile loop nest.
  // In particular, the iteration argument of the outermost tile loop needs to
  // be set to the producer output instead of the producer result and `clonedOp`
  // shall use the existing `sliceOp` result instead of the tiled producer
  // output operand.
  if (iterArg) {
    OpOperand *outputOperand =
        producerOp.getOutputOperand(producerResult.getResultNumber());
    iterArg->set(outputOperand->get());
    tiledOperands[outputOperand->getOperandNumber()] = sliceOp.getResult();
  }

  // Clone the producer using the tiled producer operands.
  TypeRange resultTypes = ValueRange(tiledOperands)
                              .take_back(producerOp.getNumOutputs())
                              .getTypes();
  LinalgOp clonedOp = producerOp.clone(b, loc, resultTypes, tiledOperands);

  // Shift all IndexOp results by the tile offset.
  addTileLoopIvsToIndexOpResults(b, clonedOp, allIvs);

  return clonedOp;
}

//===----------------------------------------------------------------------===//
// TileLoopNest specific helpers.
//===----------------------------------------------------------------------===//

bool TileLoopNest::isEmpty() { return tileLoopOps.empty(); }

bool TileLoopNest::isValid() {
  // Check if `rootOp` has been tiled at least once.
  if (isEmpty() || tiledRootAndFusedOpsLoops.count(rootOp) == 0)
    return false;

  // Check if the number of loop operations and dimensions match.
  if (tileLoopOps.size() != tiledRootAndFusedOpsLoops[rootOp].size())
    return false;

  // Check if the innermost tile loop is the parent of `tiledOp`.
  if (rootOp->getParentOp() != tileLoopOps.back())
    return false;

  // Check if the tile loops are directly nested.
  return std::adjacent_find(tileLoopOps.begin(), tileLoopOps.end(),
                            [](Operation *op1, Operation *op2) {
                              return op1 != op2->getParentOp();
                            }) == tileLoopOps.end();
}

SmallVector<BlockArgument> TileLoopNest::getTiedBBArgs(BlockArgument bbArg) {
  assert(bbArg && "expect the block argument to be non-zero");
  SmallVector<BlockArgument> bbArgs;

  // Search all tile loop block arguments from inner to outer.
  for (auto tileLoop : reverse(tileLoopOps)) {
    if (bbArg.getOwner()->getParentOp() != tileLoop)
      return {};
    bbArgs.push_back(bbArg);
    OpOperand *iterArg = &tileLoop.getOpOperandForRegionIterArg(bbArg);
    bbArg = iterArg->get().dyn_cast<BlockArgument>();
  }

  // Reverse the block arguments to order them from outer to inner.
  return {bbArgs.rbegin(), bbArgs.rend()};
}

OpOperand *TileLoopNest::getTiedIterArg(BlockArgument bbArg) {
  // Search all block arguments and return the matching iteration argument.
  SmallVector<BlockArgument> bbArgs = getTiedBBArgs(bbArg);
  if (bbArgs.size() != tileLoopOps.size())
    return nullptr;
  return &tileLoopOps.front().getOpOperandForRegionIterArg(bbArgs.front());
}

bool TileLoopNest::hasOtherUses(BlockArgument bbArg,
                                tensor::ExtractSliceOp sliceOp) {
  // Check the innermost block argument is either used by the ExtractSliceOp
  // `sliceOp`, the matching InsertSliceOp, or by a DimOp. Handle other uses
  // conservatively.
  for (Operation *op : bbArg.getUsers()) {
    if (!isa<tensor::DimOp, tensor::InsertSliceOp, tensor::ExtractSliceOp>(op))
      return false;
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (extractSliceOp != sliceOp)
        return false;
    }
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      SetVector<Operation *> backwardSlice;
      getBackwardSlice(insertSliceOp.source(), &backwardSlice,
                       [](Operation *op) {
                         return isa<LinalgOp, tensor::InsertSliceOp>(op);
                       });
      if (backwardSlice.empty() || backwardSlice.front() != sliceOp)
        return false;
    }
  }

  // Check the block arguments, except for the innermost one, have one use.
  SmallVector<BlockArgument> bbArgs = getTiedBBArgs(bbArg);
  return !all_of(bbArgs, [&](BlockArgument bbArg) {
    return bbArg.hasOneUse() || bbArg == bbArgs.back();
  });
}

LogicalResult TileLoopNest::tileRootOp(
    OpBuilder &b, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange,
    Optional<LinalgLoopDistributionOptions> tileDistribution) {
  // Exit if all tile sizes are zero.
  if (tileSizes.size() == static_cast<size_t>(count(tileSizes, 0)))
    return success();

  // Tile the root operation.
  LinalgTilingOptions tilingOptions;
  tilingOptions = tilingOptions
                      .setInterchange(SmallVector<unsigned>(
                          tileInterchange.begin(), tileInterchange.end()))
                      .setTileSizes(tileSizes)
                      .setLoopType(LinalgTilingLoopType::Loops);
  if (tileDistribution)
    tilingOptions =
        tilingOptions.setDistributionOptions(tileDistribution.getValue());

  // TODO: Propagate RewriterBase everywhere.
  IRRewriter rewriter(b);
  FailureOr<TiledLinalgOp> tiledRootOp =
      tileLinalgOp(rewriter, rootOp, tilingOptions);

  // Exit if tiling the root operation fails.
  if (failed(tiledRootOp))
    return failure();

  // Replace all uses of the root operation if it has been tiled before. All
  // uses of the original untiled root operation are updated by the calling pass
  // or pattern.
  if (!isEmpty())
    rootOp->replaceAllUsesWith(tiledRootOp->tensorResults);
  fusedOps.push_back(rootOp);
  for (auto result : llvm::enumerate(rootOp->getResults()))
    replacements[result.value()] = tiledRootOp->tensorResults[result.index()];

  // Transfer the stored `rootOp` loop dimensions if it has been tiled before.
  if (tiledRootAndFusedOpsLoops.count(rootOp) != 0) {
    tiledRootAndFusedOpsLoops[tiledRootOp->op] =
        tiledRootAndFusedOpsLoops[rootOp];
  }

  // Update the root operation and append the loops and tile loop dimensions.
  rootOp = tiledRootOp->op;
  tileLoopOps.append(tiledRootOp->loops.begin(), tiledRootOp->loops.end());
  for (const auto &en : enumerate(tileSizes)) {
    // Copy only the tiled loop dimensions with non-zero tile size.
    if (en.value() == 0)
      continue;
    tiledRootAndFusedOpsLoops[rootOp].push_back(tileInterchange[en.index()]);
  }
  assert(isValid() && "expect tile loop nest to be valid after tiling");
  return success();
}

void TileLoopNest::yieldFusedValues(OpBuilder &b, LinalgOp untiledOp,
                                    LinalgOp tiledOp,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<OpFoldResult> strides) {
  if (tileLoopOps.empty())
    return;

  SmallVector<Value> newIterOperands;
  SmallVector<Value> yieldedResults;
  SmallVector<Value> returnedFusedOpResults;
  for (auto result : llvm::enumerate(untiledOp->getResults())) {
    // Check if any of the uses are outside the tile loop nest. Dim
    // uses are not treated as "real" uses, further the dim ops could be used
    // to compute the bounds of the loop itself. They are expected to be
    // resolved using `ReifyRankedShapedTypeOpInterface`.
    if (llvm::all_of(result.value().getUsers(), [&](Operation *user) {
          return tileLoopOps.front()->isAncestor(user) ||
                 isa<tensor::DimOp>(user) || llvm::is_contained(fusedOps, user);
          ;
        }))
      continue;

    unsigned resultNumber = result.value().getResultNumber();
    // 1. Replace all existing uses of untiled op in the iter_args of the tile
    //    loops with the `outs` of the untiled operation.
    OpOperand *untiledOpOutOperand = untiledOp.getOutputOperand(resultNumber);
    tileLoopOps.front()->replaceUsesOfWith(result.value(),
                                           untiledOpOutOperand->get());

    // 2. The `outs` ops of the untiled op are the "new" iter operands to be
    //    added.
    newIterOperands.push_back(untiledOpOutOperand->get());
    yieldedResults.push_back(tiledOp->getResult(resultNumber));
    returnedFusedOpResults.push_back(result.value());
  }

  // The inner most loop body needs to be modified to add `tensor.insert_slice`s
  // to reconstruct the full tensor using the values produced from the tiled
  // ops.
  scf::ForOp outerMostLoop = tileLoopOps.front();
  scf::ForOp innerMostLoop = tileLoopOps.back();
  NewYieldValueFn innerFn = [&](OpBuilder &innerBuilder, Location innerLoc,
                                ArrayRef<BlockArgument> innerNewBBArgs) {
    SmallVector<Value> newYieldVals;
    for (auto yieldedResult : llvm::enumerate(yieldedResults)) {
      newYieldVals.push_back(innerBuilder.create<tensor::InsertSliceOp>(
          tiledOp->getLoc(), yieldedResult.value(),
          innerNewBBArgs[yieldedResult.index()], offsets, sizes, strides));
    }
    return newYieldVals;
  };
  scf::ForOp newInnerMostLoop =
      replaceLoopWithNewYields(b, innerMostLoop, newIterOperands, innerFn);
  tileLoopOps.back() = newInnerMostLoop;

  // Propagate the newly yielded values back up the loop nest.
  for (unsigned loopDepth :
       llvm::reverse(llvm::seq<unsigned>(0, tileLoopOps.size() - 1))) {
    NewYieldValueFn fn = [&](OpBuilder &innerBuilder, Location innerLoc,
                             ArrayRef<BlockArgument> innerNewBBArgs) {
      return llvm::to_vector(
          llvm::map_range(tileLoopOps[loopDepth + 1].getResults().take_back(
                              newIterOperands.size()),
                          [](OpResult r) -> Value { return r; }));
    };
    tileLoopOps[loopDepth] = replaceLoopWithNewYields(b, tileLoopOps[loopDepth],
                                                      newIterOperands, fn);
  }

  // The `replacements` uses values returned by the tile loop nest. Update these
  // to use values from the new loop nests created here. Build a map from result
  // of the previous outermost loop to the new outermost loop.
  llvm::SmallDenseMap<Value, Value> resultValRemap;
  scf::ForOp newOuterMostLoop = tileLoopOps.front();
  for (auto origResult : llvm::enumerate(outerMostLoop->getResults())) {
    resultValRemap[origResult.value()] =
        newOuterMostLoop->getResult(origResult.index());
  }
  for (auto &replacement : replacements) {
    auto newReplacement = resultValRemap.lookup(replacement.second);
    if (newReplacement)
      replacement.second = newReplacement;
  }
  for (auto fusedOpResult : llvm::enumerate(returnedFusedOpResults)) {
    replacements[fusedOpResult.value()] = newOuterMostLoop.getResult(
        outerMostLoop.getNumResults() + fusedOpResult.index());
  }
}

FailureOr<LinalgOp> TileLoopNest::fuseProducer(OpBuilder &b,
                                               OpOperand *consumerOpOperand,
                                               bool returnFusedOpValues) {
  // Check if the consumer has been tiled before. For example, it may not have
  // been tiled if the outermost tile loop is a reduction loop.
  if (tiledRootAndFusedOpsLoops.count(consumerOpOperand->getOwner()) == 0)
    return failure();

  assert(this->isValid() &&
         "expect the tile loop nest to satisfy all invariants");

  // Check the tile loop nest is non-empty.
  if (isEmpty())
    return failure();

  // Check `consumerOpOperand` is defined by an ExtractSliceOp.
  auto sliceOp =
      consumerOpOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure();

  // Check `sliceOp` and `consumerOp` are in the same block.
  LinalgOp consumerOp = consumerOpOperand->getOwner();
  if (sliceOp->getBlock() != rootOp->getBlock() ||
      consumerOp->getBlock() != rootOp->getBlock())
    return failure();

  // Check `consumerOpOperand` is not shape-only to avoid fusion if the data is
  // not used by the `consumerOp` computation.
  BlockArgument bbArg = consumerOp.getTiedBlockArgument(consumerOpOperand);
  if (bbArg.getUses().empty())
    return failure();

  // Check if the producer is a LinalgOp possibly passed by iteration argument.
  OpOperand *iterArg = nullptr;
  auto producerResult = sliceOp.source().dyn_cast<OpResult>();
  if (auto bbArg = sliceOp.source().dyn_cast<BlockArgument>()) {
    if (bbArg.getParentBlock() != sliceOp->getBlock())
      return failure();
    iterArg = getTiedIterArg(bbArg);
    // Check the iteration argument may be used to pass in the producer output.
    if (!iterArg || hasOtherUses(bbArg, sliceOp))
      return failure();
    producerResult = iterArg->get().dyn_cast<OpResult>();
  }
  if (!producerResult)
    return failure();
  LinalgOp producer = dyn_cast<LinalgOp>(producerResult.getOwner());
  if (!producer)
    return failure();

  // Compute the tiled producer slice dimensions given the tiled consumer loops.
  SmallVector<int64_t> tiledSliceDimIndices = getTiledSliceDims(
      consumerOpOperand, tiledRootAndFusedOpsLoops[consumerOp]);
  if (tiledSliceDimIndices.empty())
    return failure();

  // Compute the tiled producer loop indices.
  SmallVector<int64_t> tiledProducerLoopIndices =
      getTiledProducerLoops(producerResult, tiledSliceDimIndices);

  // Tile the producer operands and clone the producer in place of `sliceOp`.
  LinalgOp clonedOp =
      getTiledProducer(b, producerResult, sliceOp, tiledSliceDimIndices,
                       tiledProducerLoopIndices, iterArg);
  tiledRootAndFusedOpsLoops[clonedOp] = tiledProducerLoopIndices;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(clonedOp);

  // Cast the `clonedOp` result to gap type mismatches before canonicalization.
  Type consumerOperandType = consumerOpOperand->get().getType();
  Value newResult = clonedOp->getResult(producerResult.getResultNumber());
  if (newResult.getType() != consumerOperandType) {
    newResult = b.create<tensor::CastOp>(producerResult.getLoc(),
                                         consumerOperandType, newResult);
  }

  // Replace the `sliceOp` uses except for the `clonedOp` output uses.
  sliceOp.getResult().replaceAllUsesExcept(newResult, clonedOp);

  if (returnFusedOpValues) {
    fusedOps.push_back(producer);
    yieldFusedValues(b, producer, clonedOp, sliceOp.getMixedOffsets(),
                     sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  }
  return clonedOp;
}

Value TileLoopNest::getUntiledOpResultReplacement(Value v) {
  return replacements.lookup(v);
}

SmallVector<LinalgOp> TileLoopNest::getAllTiledAndFusedOps() {
  SmallVector<LinalgOp> result;
  for (const auto &kvp : tiledRootAndFusedOpsLoops) {
    auto linalgOp = dyn_cast<LinalgOp>(kvp.getFirst());
    assert(linalgOp &&
           "expect all tiled and fused operations are linalg operations");
    result.push_back(linalgOp);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Tile and fuse entry-points.
//===----------------------------------------------------------------------===//

FailureOr<TileLoopNest> mlir::linalg::tileConsumerAndFuseProducers(
    OpBuilder &b, LinalgOp consumerOp, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange,
    const Optional<LinalgLoopDistributionOptions> &tileDistribution,
    bool returnFusedOpValues) {
  assert(tileSizes.size() == tileInterchange.size() &&
         "expect the number of tile sizes and interchange dims to match");
  assert(isPermutation(tileInterchange) &&
         "expect tile interchange is a permutation");

  // Create an empty tile loop nest.
  TileLoopNest tileLoopNest(consumerOp);

  // Search the number of outer parallel loops to separate them from possible
  // inner reduction dimensions.
  SmallVector<StringAttr> iterTypes =
      llvm::to_vector<6>(consumerOp.iterator_types().getAsRange<StringAttr>());
  applyPermutationToVector(iterTypes, tileInterchange);
  auto *it = find_if(iterTypes, [&](StringAttr iterType) {
    return !isParallelIterator(iterType);
  });
  int64_t split = std::distance(iterTypes.begin(), it);

  // Helper to fuse the producers greedily using a queue of fusion candidates.
  auto fuseProducersGreedily = [&](ArrayRef<OpOperand *> operands,
                                   bool returnFusedOpValues) {
    SmallVector<OpOperand *> candidates(operands.begin(), operands.end());
    while (!candidates.empty()) {
      FailureOr<LinalgOp> fusedProducer = tileLoopNest.fuseProducer(
          b, candidates.pop_back_val(), returnFusedOpValues);
      if (failed(fusedProducer))
        continue;
      candidates.append(fusedProducer->getInputAndOutputOperands());
    }
  };

  // Tile the outer parallel loops and fuse the output operands.
  SmallVector<int64_t> outerTileSizes;
  outerTileSizes.append(tileSizes.begin(), tileSizes.begin() + split);
  outerTileSizes.append(tileSizes.size() - split, 0);
  if (failed(tileLoopNest.tileRootOp(b, outerTileSizes, tileInterchange,
                                     tileDistribution)))
    return failure();
  if (returnFusedOpValues) {
    fuseProducersGreedily(tileLoopNest.getRootOp().getInputAndOutputOperands(),
                          true);
  } else {
    fuseProducersGreedily(tileLoopNest.getRootOp().getOutputOperands(), false);
  }

  // Tile the remaining loops and fuse the input operands.
  SmallVector<int64_t> innerTileSizes;
  innerTileSizes.append(split, 0);
  innerTileSizes.append(tileSizes.begin() + split, tileSizes.end());
  if (failed(tileLoopNest.tileRootOp(b, innerTileSizes, tileInterchange,
                                     tileDistribution)))
    return failure();
  fuseProducersGreedily(tileLoopNest.getRootOp().getInputOperands(), false);

  // Exit if the tile loop nest is empty since all tile sizes are zero.
  if (tileLoopNest.isEmpty())
    return failure();

  return tileLoopNest;
}
