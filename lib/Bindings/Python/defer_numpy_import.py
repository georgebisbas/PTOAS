#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Defer upstream MLIR's eager numpy import in the installed ``ptoas/mlir/ir.py``.

MLIR's Python bindings import numpy at module scope to register the optional
``DenseElementsAttr`` builders (``F64ElementsAttr``, ``I32ElementsAttr``,
``I64ElementsAttr``, ``IndexElementsAttr``). ptoas is a short-lived CLI -- one
process per compiled kernel -- so that import sits on the critical path of every
invocation: it costs ~85 ms of the ~110 ms Python start-up on the release
wheels, although the builders are not needed unless one of those attributes is
built from Python.

This script rewrites the module so the builders stay registered and numpy is
imported on first use. It runs from the ``PTOAS_Python`` install component
(see ``lib/Bindings/Python/CMakeLists.txt``) and is idempotent: a file whose
import is already deferred, or one that no longer imports numpy at all, is left
untouched. A module-scope import in a shape this script does not recognise
exits with code 1, so a silent start-up regression cannot ship;
``docker/validate_wheel_payload.py`` re-checks the built wheel the same way
without needing a build.

Usage::

    defer_numpy_import.py <prefix>/ptoas/mlir/ir.py
    defer_numpy_import.py --check <prefix>/ptoas/mlir/ir.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EAGER_MARKER = "try:\n    import numpy as np\n"
BUILDER_DEF = re.compile(r"^    def _\w+Attr\(")
MODULE_SCOPE_IMPORT = re.compile(r"^(import numpy|    import numpy)")


def _defer_builder_imports(block: str) -> str:
    """Move numpy into each builder that needs it inside an eager-import block."""
    lines = block.splitlines(keepends=True)
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        if BUILDER_DEF.match(line) is None:
            continue
        # The builder body is the following 8-space indented region (blank
        # lines included); the next decorator or blank-then-decorator line
        # ends it.
        body: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            if candidate.startswith("        ") or candidate.strip() == "":
                body.append(candidate)
                index += 1
            else:
                break
        if any("np." in item for item in body):
            out.append("        import numpy as np\n")
        out.extend(body)
    return "".join(out)


def defer_module_import(source: str) -> tuple[str, str]:
    """Return ``(rewritten_source, status)`` for a staged MLIR ``ir.py``.

    Statuses: ``"deferred"`` (rewritten), ``"already-deferred"``,
    ``"no-numpy"`` (upstream no longer imports it) and ``"unrecognised"``
    (module-scope import in an unexpected shape -- the caller must fail).
    """
    if EAGER_MARKER not in source:
        if "import numpy" not in source:
            return source, "no-numpy"
        if MODULE_SCOPE_IMPORT.search(source):
            return source, "unrecognised"
        return source, "already-deferred"
    start = source.index(EAGER_MARKER)
    end = source.find("\nexcept ImportError:", start)
    if end < 0:
        return source, "unrecognised"
    head = source[:start]
    block = source[start:end].replace(EAGER_MARKER, "try:\n", 1)
    tail = source[end:]
    return head + _defer_builder_imports(block) + tail, "deferred"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="installed ptoas/mlir/ir.py")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the import is deferred; never modify the file",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    source = path.read_text(encoding="utf-8")
    rewritten, status = defer_module_import(source)
    if args.check:
        if status in ("deferred", "unrecognised"):
            print(
                f"error: {path} still imports numpy at module scope ({status})",
                file=sys.stderr,
            )
            return 1
        print(f"{path}: numpy import is deferred ({status})")
        return 0

    if status == "deferred":
        path.write_text(rewritten, encoding="utf-8")
        print(f"{path}: deferred numpy import into the DenseElementsAttr builders")
        return 0
    if status in ("already-deferred", "no-numpy"):
        print(f"{path}: numpy import already deferred ({status}); nothing to do")
        return 0
    print(
        f"error: {path} imports numpy at module scope in an unrecognised shape; "
        "update defer_numpy_import.py for the new upstream layout",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
