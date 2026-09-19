from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("pip-audit.json")
    if not path.is_file() or path.stat().st_size == 0:
        print(json.dumps({"researchDependencyFindings": []}, sort_keys=True))
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    for dependency in data.get("dependencies", []):
        vulnerabilities = dependency.get("vulns", [])
        if not vulnerabilities:
            continue
        findings.append(
            {
                "name": dependency.get("name"),
                "version": dependency.get("version"),
                "advisories": [
                    {
                        "id": item.get("id"),
                        "fixVersions": item.get("fix_versions", []),
                    }
                    for item in vulnerabilities
                ],
            }
        )
    print(json.dumps({"researchDependencyFindings": findings}, sort_keys=True))


if __name__ == "__main__":
    main()
