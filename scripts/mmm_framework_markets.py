"""mmm-framework's generated worlds, exported once as files, outside the bench's environment.

mmm-framework 1.5.0's wheel ships two data sets as files; the rest of its synthetic worlds come from
its generator, ``mmm_framework.synth``. This script runs that generator as a black box, in the
environment ``scripts/mmm_framework_arm.txt`` pins, by the calls its docstrings document and no
other: ``generate_mff(name)`` for each name given, at the factory's default seed and length, which
returns a master flat file as a DataFrame and the world's answer key as a JSON-safe dict. It writes
each as ``<name>_mff.csv`` (the DataFrame's ``to_csv(index=False)``) and ``<name>_truth.json``
(``json.dumps(key, indent=2)`` and a newline), and ``MANIFEST.json``: the tool's version, the call,
and each file's size and SHA-256, which the bench reads the files by
(:mod:`causaldyn_bench.scorecard.mmm_framework`). Run offline, one process::

    unshare -rn VENV/bin/python -I scripts/mmm_framework_markets.py OUT_DIR NAME [NAME ...]

The generator prints nothing the files depend on; a warning it raises is recorded by its type and
message, with how often.
"""

import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path


def main():
    out, names = Path(sys.argv[1]), sys.argv[2:]
    import mmm_framework  # ty: ignore[unresolved-import]
    from mmm_framework.synth import generate_mff  # ty: ignore[unresolved-import]

    out.mkdir(parents=True, exist_ok=True)
    worlds = {}
    for name in names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame, key = generate_mff(name)
        files = {
            "mff.csv": frame.to_csv(index=False).encode(),
            "truth.json": (json.dumps(key, indent=2) + "\n").encode(),
        }
        for kind, data in files.items():
            (out / f"{name}_{kind}").write_bytes(data)
        worlds[name] = {
            kind: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for kind, data in files.items()
        }
        worlds[name]["warnings"] = dict(
            Counter(f"{w.category.__name__}: {w.message}" for w in caught)
        )
    manifest = {
        "tool": "mmm-framework",
        "version": mmm_framework.__version__,
        "call": "mmm_framework.synth.generate_mff(name)",
        "worlds": worlds,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
