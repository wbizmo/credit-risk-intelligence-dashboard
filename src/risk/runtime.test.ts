import rawArtifact from "../../model/artifacts/crix-monoboost-v2.json";
import { describe, expect, it } from "vitest";
import {
  compileModelArtifact,
  evaluateCompiledBaseline,
  evaluateCompiledMargin,
  evaluateMarginWithFeatureOverride,
  evaluateReferenceMargin,
  type ModelArtifact,
} from "./runtime";

const cloneArtifact = (): ModelArtifact => structuredClone(rawArtifact) as ModelArtifact;

describe("compiled CRIX runtime", () => {
  it("compiles the committed artifact into a frozen execution plan", () => {
    const model = compileModelArtifact(cloneArtifact());
    expect(model.trees.length).toBe(rawArtifact.trees.length);
    expect(model.featureNames).toEqual(rawArtifact.featureNames);
    expect(model.maxTreeDepth).toBeLessThanOrEqual(64);
    expect(Object.isFrozen(model)).toBe(true);
    expect(Object.isFrozen(model.trees)).toBe(true);
    expect(Object.isFrozen(model.trees[0])).toBe(true);
    expect(Object.isFrozen(model.trees[0]?.threshold)).toBe(true);
  });

  it("fails closed on malformed child indexes", () => {
    const artifact = cloneArtifact();
    artifact.trees[0]!.left[0] = artifact.trees[0]!.left.length + 10;
    expect(() => compileModelArtifact(artifact)).toThrow(/child index/);
  });

  it("fails closed on out-of-range feature indexes", () => {
    const artifact = cloneArtifact();
    artifact.trees[0]!.feature[0] = artifact.featureNames.length;
    expect(() => compileModelArtifact(artifact)).toThrow(/feature index/);
  });

  it("fails closed on non-finite node values", () => {
    const artifact = cloneArtifact();
    artifact.trees[0]!.threshold[0] = Number.NaN;
    expect(() => compileModelArtifact(artifact)).toThrow(/non-finite/);
  });

  it("fails closed on invalid challenger scales", () => {
    const artifact = cloneArtifact();
    artifact.challenger.scales[0] = 0;
    expect(() => compileModelArtifact(artifact)).toThrow(/scale/);
  });

  it("fails closed on inconsistent tree array lengths", () => {
    const artifact = cloneArtifact();
    artifact.trees[0]!.right.pop();
    expect(() => compileModelArtifact(artifact)).toThrow(/inconsistent array lengths/);
  });

  it("fails closed on cyclic tree structure", () => {
    const artifact = cloneArtifact();
    artifact.trees[0]!.left[0] = 0;
    expect(() => compileModelArtifact(artifact)).toThrow(/cycle/);
  });

  it("matches the reference evaluator across a deterministic vector corpus", () => {
    const artifact = cloneArtifact();
    const model = compileModelArtifact(artifact);
    for (let index = 0; index < 250; index += 1) {
      const vector = [
        0.02 + ((index * 17) % 34) / 100,
        0.03 + ((index * 11) % 44) / 100,
        660 + ((index * 13) % 140),
        0.5 + ((index * 7) % 20) / 2,
      ];
      expect(evaluateCompiledMargin(model, vector)).toBeCloseTo(evaluateReferenceMargin(artifact, vector), 14);
    }
  });

  it("sparse overrides match a full compiled re-evaluation", () => {
    const model = compileModelArtifact(cloneArtifact());
    const vector = [0.28, 24_000 / 85_000, 720, 5];
    const baseline = evaluateCompiledBaseline(model, vector);

    for (let featureIndex = 0; featureIndex < model.featureNames.length; featureIndex += 1) {
      const override = model.reference[featureIndex]!;
      const overridden = [...vector];
      overridden[featureIndex] = override;
      const sparseMargin = evaluateMarginWithFeatureOverride(model, baseline, vector, featureIndex, override);
      expect(sparseMargin).toBeCloseTo(evaluateCompiledMargin(model, overridden), 12);
    }
  });

  it("indexes an unused feature without explanation tree work", () => {
    const artifact = cloneArtifact();
    artifact.featureNames.push("unusedFeature");
    artifact.monotoneConstraints.push(0);
    artifact.challenger.coefficients.push(0);
    artifact.challenger.means.push(0);
    artifact.challenger.scales.push(1);
    artifact.reference.unusedFeature = 0;
    artifact.trainingBounds.unusedFeature = { p01: -1, p99: 1 };
    const model = compileModelArtifact(artifact);
    expect(model.treesByFeature[model.featureNames.length - 1]).toEqual([]);
  });

  it("indexes every tree when all trees use the same feature", () => {
    const artifact = cloneArtifact();
    for (const tree of artifact.trees) {
      for (let node = 0; node < tree.left.length; node += 1) {
        if (tree.left[node] !== -1) tree.feature[node] = 0;
      }
    }
    const model = compileModelArtifact(artifact);
    expect(model.treesByFeature[0]?.length).toBe(model.trees.length);
  });
});
