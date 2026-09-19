import rawArtifact from "../../model/artifacts/crix-monoboost-v2.json";
import { verifyRuntimeArtifactManifest } from "./registry";

export interface ModelTree {
  left: number[];
  right: number[];
  feature: number[];
  threshold: number[];
  defaultLeft: number[];
}

export interface ModelArtifact {
  schemaVersion: number;
  name: string;
  version: string;
  trainedAt: string;
  target: { name: string; definition: string; horizon: string };
  featureNames: string[];
  monotoneConstraints: number[];
  baseScore: number;
  calibration: { method: string; slope: number; intercept: number };
  challenger: {
    name: string;
    intercept: number;
    coefficients: number[];
    means: number[];
    scales: number[];
  };
  trees: ModelTree[];
  metrics: Record<string, number>;
  diagnostics: {
    calibration: Array<{ predicted: number; observed: number; count: number }>;
    roc: Array<{ fpr: number; tpr: number }>;
    featureImportance: Array<{ feature: string; gain: number }>;
  };
  reference: Record<string, number>;
  trainingBounds: Record<string, { p01: number; p99: number }>;
  training: Record<string, unknown>;
}

export interface CompiledTree {
  left: readonly number[];
  right: readonly number[];
  feature: readonly number[];
  threshold: readonly number[];
  defaultLeft: readonly number[];
}

export interface CompiledModel {
  baseMargin: number;
  featureNames: readonly string[];
  featureIndex: Readonly<Record<string, number>>;
  calibrationSlope: number;
  calibrationIntercept: number;
  challengerIntercept: number;
  challengerCoefficients: readonly number[];
  challengerMeans: readonly number[];
  challengerScales: readonly number[];
  reference: readonly number[];
  lowerBounds: readonly number[];
  upperBounds: readonly number[];
  trees: readonly CompiledTree[];
  treesByFeature: readonly (readonly number[])[];
  maxTreeDepth: number;
}

export interface ComplexityCounters {
  championTreeVisits: number;
  championNodeVisits: number;
  explanationTreeVisits: number;
  explanationNodeVisits: number;
  challengerFeatureOps: number;
}

export interface BaselineEvaluation {
  margin: number;
  treeContributions: readonly number[];
}

const MAX_TREE_DEPTH = 64;
const artifact = rawArtifact as ModelArtifact;
let runtimeModel: CompiledModel | undefined;
let runtimeError: Error | undefined;

const copyNumbers = (values: readonly number[]): readonly number[] => Array.from(values);
const logit = (value: number) => Math.log(value / (1 - value));

function assertFinite(value: number, message: string): void {
  if (!Number.isFinite(value)) throw new Error(message);
}

function validateTree(tree: ModelTree, treeIndex: number, featureCount: number): { depth: number; usedFeatures: Set<number> } {
  const nodeCount = tree.left.length;
  if (nodeCount === 0) throw new Error(`Model tree ${treeIndex} has no nodes`);
  const lengths = [tree.right.length, tree.feature.length, tree.threshold.length, tree.defaultLeft.length];
  if (lengths.some((length) => length !== nodeCount)) throw new Error(`Model tree ${treeIndex} has inconsistent array lengths`);

  for (let node = 0; node < nodeCount; node += 1) {
    const left = tree.left[node]!;
    const right = tree.right[node]!;
    const feature = tree.feature[node]!;
    const threshold = tree.threshold[node]!;
    const defaultLeft = tree.defaultLeft[node]!;

    if (!Number.isInteger(left) || !Number.isInteger(right)) throw new Error(`Model tree ${treeIndex} has a non-integer child index`);
    if (!Number.isInteger(feature)) throw new Error(`Model tree ${treeIndex} has a non-integer feature index`);
    if (feature < 0 || feature >= featureCount) throw new Error(`Model tree ${treeIndex} has an out-of-range feature index`);
    if (defaultLeft !== 0 && defaultLeft !== 1) throw new Error(`Model tree ${treeIndex} has an invalid default-left flag`);
    assertFinite(threshold, `Model tree ${treeIndex} has a non-finite threshold/leaf`);

    const leaf = left === -1 || right === -1;
    if (leaf) {
      if (left !== -1 || right !== -1) throw new Error(`Model tree ${treeIndex} has a partially-defined leaf`);
      continue;
    }

    if (left < 0 || right < 0 || left >= nodeCount || right >= nodeCount) {
      throw new Error(`Model tree ${treeIndex} has an out-of-range child index`);
    }
  }

  const state = new Array<number>(nodeCount).fill(0);
  const depthMemo = new Array<number>(nodeCount).fill(0);
  const reachable = new Set<number>();
  const usedFeatures = new Set<number>();

  const visit = (node: number): number => {
    if (state[node] === 1) throw new Error(`Model tree ${treeIndex} contains a cycle`);
    if (state[node] === 2) return depthMemo[node]!;
    state[node] = 1;
    reachable.add(node);

    const left = tree.left[node]!;
    let depth = 1;
    if (left !== -1) {
      const right = tree.right[node]!;
      const feature = tree.feature[node]!;
      usedFeatures.add(feature);
      depth = 1 + Math.max(visit(left), visit(right));
    }

    state[node] = 2;
    depthMemo[node] = depth;
    return depth;
  };

  const depth = visit(0);
  if (reachable.size !== nodeCount) throw new Error(`Model tree ${treeIndex} contains unreachable nodes`);
  if (depth > MAX_TREE_DEPTH) throw new Error(`Model tree ${treeIndex} exceeds the traversal bound`);
  return { depth, usedFeatures };
}

export function compileModelArtifact(source: ModelArtifact): CompiledModel {
  if (source.schemaVersion !== 2 || !source.name || !source.version || source.trees.length === 0) {
    throw new Error("Invalid CRIX model artifact envelope");
  }

  const featureNames = Array.from(source.featureNames);
  const featureCount = featureNames.length;
  if (featureCount === 0 || source.monotoneConstraints.length !== featureCount) {
    throw new Error("Model feature contract is inconsistent");
  }
  if (new Set(featureNames).size !== featureCount || featureNames.some((name) => !name)) {
    throw new Error("Model feature names must be unique and non-empty");
  }

  assertFinite(source.baseScore, "Model base score is non-finite");
  if (source.baseScore <= 0 || source.baseScore >= 1) throw new Error("Model base score must be strictly between zero and one");
  assertFinite(source.calibration.slope, "Model calibration slope is non-finite");
  assertFinite(source.calibration.intercept, "Model calibration intercept is non-finite");

  const challenger = source.challenger;
  if (
    challenger.coefficients.length !== featureCount ||
    challenger.means.length !== featureCount ||
    challenger.scales.length !== featureCount
  ) {
    throw new Error("Challenger artifact dimensions do not match champion feature contract");
  }
  assertFinite(challenger.intercept, "Challenger intercept is non-finite");
  for (let index = 0; index < featureCount; index += 1) {
    assertFinite(challenger.coefficients[index]!, "Challenger coefficient is non-finite");
    assertFinite(challenger.means[index]!, "Challenger mean is non-finite");
    const scale = challenger.scales[index]!;
    assertFinite(scale, "Challenger scale is non-finite");
    if (scale <= 0) throw new Error("Challenger scale must be greater than zero");
  }

  const featureIndex: Record<string, number> = Object.create(null) as Record<string, number>;
  const reference: number[] = [];
  const lowerBounds: number[] = [];
  const upperBounds: number[] = [];
  for (let index = 0; index < featureCount; index += 1) {
    const feature = featureNames[index]!;
    featureIndex[feature] = index;
    const bounds = source.trainingBounds[feature];
    const ref = source.reference[feature];
    if (!bounds || ref === undefined) throw new Error(`Missing explanation metadata for feature: ${feature}`);
    assertFinite(bounds.p01, `Non-finite lower bound for feature: ${feature}`);
    assertFinite(bounds.p99, `Non-finite upper bound for feature: ${feature}`);
    assertFinite(ref, `Non-finite reference for feature: ${feature}`);
    if (bounds.p01 > bounds.p99 || ref < bounds.p01 || ref > bounds.p99) {
      throw new Error(`Invalid explanation bounds/reference for feature: ${feature}`);
    }
    lowerBounds.push(bounds.p01);
    upperBounds.push(bounds.p99);
    reference.push(ref);
  }

  const treesByFeature = Array.from({ length: featureCount }, () => [] as number[]);
  let maxTreeDepth = 0;
  const trees = source.trees.map((tree, treeIndex) => {
    const validation = validateTree(tree, treeIndex, featureCount);
    maxTreeDepth = Math.max(maxTreeDepth, validation.depth);
    for (const feature of validation.usedFeatures) treesByFeature[feature]!.push(treeIndex);
    return {
      left: copyNumbers(tree.left),
      right: copyNumbers(tree.right),
      feature: copyNumbers(tree.feature),
      threshold: copyNumbers(tree.threshold),
      defaultLeft: copyNumbers(tree.defaultLeft),
    };
  });

  return Object.freeze({
    baseMargin: logit(source.baseScore),
    featureNames,
    featureIndex: Object.freeze(featureIndex),
    calibrationSlope: source.calibration.slope,
    calibrationIntercept: source.calibration.intercept,
    challengerIntercept: challenger.intercept,
    challengerCoefficients: copyNumbers(challenger.coefficients),
    challengerMeans: copyNumbers(challenger.means),
    challengerScales: copyNumbers(challenger.scales),
    reference: copyNumbers(reference),
    lowerBounds: copyNumbers(lowerBounds),
    upperBounds: copyNumbers(upperBounds),
    trees,
    treesByFeature,
    maxTreeDepth,
  });
}

export function evaluateCompiledMargin(model: CompiledModel, vector: readonly number[]): number {
  if (vector.length !== model.featureNames.length) throw new Error("Feature vector length does not match compiled model");
  let margin = model.baseMargin;

  for (const tree of model.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const featureIndex = tree.feature[node]!;
      const value = vector[featureIndex]!;
      node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
        ? tree.left[node]!
        : tree.right[node]!;
    }
    margin += tree.threshold[node]!;
  }

  if (!Number.isFinite(margin)) throw new Error("Model produced a non-finite margin");
  return margin;
}

export function evaluateCompiledBaseline(model: CompiledModel, vector: readonly number[]): BaselineEvaluation {
  if (vector.length !== model.featureNames.length) throw new Error("Feature vector length does not match compiled model");
  const treeContributions = new Array<number>(model.trees.length);
  let margin = model.baseMargin;

  for (let treeIndex = 0; treeIndex < model.trees.length; treeIndex += 1) {
    const tree = model.trees[treeIndex]!;
    let node = 0;
    while (tree.left[node] !== -1) {
      const featureIndex = tree.feature[node]!;
      const value = vector[featureIndex]!;
      node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
        ? tree.left[node]!
        : tree.right[node]!;
    }
    const contribution = tree.threshold[node]!;
    treeContributions[treeIndex] = contribution;
    margin += contribution;
  }

  if (!Number.isFinite(margin)) throw new Error("Model produced a non-finite margin");
  return { margin, treeContributions };
}

export function evaluateMarginWithFeatureOverride(
  model: CompiledModel,
  baseline: BaselineEvaluation,
  vector: readonly number[],
  featureIndex: number,
  overrideValue: number,
): number {
  if (featureIndex < 0 || featureIndex >= model.featureNames.length) throw new Error("Feature override index is out of range");
  let margin = baseline.margin;

  for (const treeIndex of model.treesByFeature[featureIndex]!) {
    const tree = model.trees[treeIndex]!;
    margin -= baseline.treeContributions[treeIndex]!;
    let node = 0;

    while (tree.left[node] !== -1) {
      const nodeFeatureIndex = tree.feature[node]!;
      const value = nodeFeatureIndex === featureIndex ? overrideValue : vector[nodeFeatureIndex]!;
      node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
        ? tree.left[node]!
        : tree.right[node]!;
    }

    margin += tree.threshold[node]!;
  }

  if (!Number.isFinite(margin)) throw new Error("Model produced a non-finite override margin");
  return margin;
}

export function evaluateReferenceMargin(source: ModelArtifact, vector: readonly number[]): number {
  let margin = logit(source.baseScore);
  for (const tree of source.trees) {
    let node = 0;
    let hops = 0;
    while (true) {
      if (hops++ > MAX_TREE_DEPTH) throw new Error("Reference model traversal limit exceeded");
      const left = tree.left[node];
      const right = tree.right[node];
      const featureIndex = tree.feature[node];
      const threshold = tree.threshold[node];
      const defaultLeft = tree.defaultLeft[node];
      if (left === undefined || right === undefined || featureIndex === undefined || threshold === undefined || defaultLeft === undefined) {
        throw new Error("Reference model contains an invalid tree node");
      }
      if (left === -1) {
        if (!Number.isFinite(threshold)) throw new Error("Reference model contains a non-finite leaf");
        margin += threshold;
        break;
      }
      const value = vector[featureIndex];
      if (value === undefined) throw new Error("Reference model feature index is out of range");
      node = (Number.isNaN(value) ? Boolean(defaultLeft) : value < threshold) ? left : right;
      if (node < 0) throw new Error("Reference model points to an invalid child node");
    }
  }
  return margin;
}

export function emptyComplexityCounters(): ComplexityCounters {
  return {
    championTreeVisits: 0,
    championNodeVisits: 0,
    explanationTreeVisits: 0,
    explanationNodeVisits: 0,
    challengerFeatureOps: 0,
  };
}

function traverseTreeCounted(tree: CompiledTree, vector: readonly number[], counters: ComplexityCounters, explanation: boolean): number {
  let node = 0;
  if (explanation) counters.explanationTreeVisits += 1;
  else counters.championTreeVisits += 1;
  while (tree.left[node] !== -1) {
    if (explanation) counters.explanationNodeVisits += 1;
    else counters.championNodeVisits += 1;
    const featureIndex = tree.feature[node]!;
    const value = vector[featureIndex]!;
    node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
      ? tree.left[node]!
      : tree.right[node]!;
  }
  if (explanation) counters.explanationNodeVisits += 1;
  else counters.championNodeVisits += 1;
  return tree.threshold[node]!;
}

function traverseTreeOverrideCounted(
  tree: CompiledTree,
  vector: readonly number[],
  overrideFeatureIndex: number,
  overrideValue: number,
  counters: ComplexityCounters,
): number {
  let node = 0;
  counters.explanationTreeVisits += 1;
  while (tree.left[node] !== -1) {
    counters.explanationNodeVisits += 1;
    const featureIndex = tree.feature[node]!;
    const value = featureIndex === overrideFeatureIndex ? overrideValue : vector[featureIndex]!;
    node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
      ? tree.left[node]!
      : tree.right[node]!;
  }
  counters.explanationNodeVisits += 1;
  return tree.threshold[node]!;
}

export function evaluateBaselineWithCounters(
  model: CompiledModel,
  vector: readonly number[],
  counters: ComplexityCounters,
): BaselineEvaluation {
  const treeContributions = new Array<number>(model.trees.length);
  let margin = model.baseMargin;
  for (let treeIndex = 0; treeIndex < model.trees.length; treeIndex += 1) {
    const contribution = traverseTreeCounted(model.trees[treeIndex]!, vector, counters, false);
    treeContributions[treeIndex] = contribution;
    margin += contribution;
  }
  return { margin, treeContributions };
}

export function evaluateOverrideWithCounters(
  model: CompiledModel,
  baseline: BaselineEvaluation,
  vector: readonly number[],
  featureIndex: number,
  overrideValue: number,
  counters: ComplexityCounters,
): number {
  let margin = baseline.margin;
  for (const treeIndex of model.treesByFeature[featureIndex]!) {
    margin -= baseline.treeContributions[treeIndex]!;
    margin += traverseTreeOverrideCounted(model.trees[treeIndex]!, vector, featureIndex, overrideValue, counters);
  }
  return margin;
}

export function getRuntimeModel(): CompiledModel {
  if (runtimeModel) return runtimeModel;
  if (runtimeError) throw runtimeError;
  try {
    if (!verifyRuntimeArtifactManifest()) throw new Error("CRIX runtime model manifest verification failed");
    runtimeModel = compileModelArtifact(artifact);
    return runtimeModel;
  } catch (error) {
    runtimeError = error instanceof Error ? error : new Error(String(error));
    throw runtimeError;
  }
}

export function verifyCompiledRuntimeModel(): boolean {
  try {
    getRuntimeModel();
    return true;
  } catch {
    return false;
  }
}

export function runtimeArtifactForBenchmark(): ModelArtifact {
  return artifact;
}
