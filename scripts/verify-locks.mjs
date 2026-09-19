import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const sha256 = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");

const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const npmLock = JSON.parse(readFileSync("package-lock.json", "utf8"));
const pyMeta = JSON.parse(readFileSync("model/requirements.lock.meta.json", "utf8"));
const pyLock = readFileSync("model/requirements.lock", "utf8");

if (npmLock.lockfileVersion !== 3) throw new Error("package-lock.json must use lockfileVersion 3.");
if (npmLock.name !== pkg.name || npmLock.version !== pkg.version) {
  throw new Error("npm lock root identity/version does not match package.json.");
}
if (npmLock.packages?.[""]?.version !== pkg.version) {
  throw new Error("npm lock root package version drift detected.");
}

const requirementsSha = sha256("model/requirements.txt");
const lockSha = sha256("model/requirements.lock");
if (pyMeta.requirementsSha256 !== requirementsSha) {
  throw new Error("model/requirements.txt changed without regenerating requirements.lock.");
}
if (pyMeta.lockSha256 !== lockSha) {
  throw new Error("model/requirements.lock hash does not match requirements.lock.meta.json.");
}
if (pyMeta.generator !== "pip-tools==7.6.1" || pyMeta.python !== "3.12") {
  throw new Error("Python lock generator metadata is not the approved v3.2 environment.");
}
if (!pyLock.includes("--hash=sha256:")) {
  throw new Error("Python research lock must contain package hashes.");
}

process.stdout.write(`${JSON.stringify({
  npm: { lockfileVersion: npmLock.lockfileVersion, packageVersion: pkg.version },
  python: {
    requirementsSha256: requirementsSha,
    lockSha256: lockSha,
    generator: pyMeta.generator,
    python: pyMeta.python,
  },
}, null, 2)}\n`);
