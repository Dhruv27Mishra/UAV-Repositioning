"""
Ensure the repository root is on ``sys.path`` so ``import rl_agent`` works when
running ``python scripts/<name>.py`` from any working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _d in [_here.parent, *_here.parent.parents]:
    if (_d / "rl_agent" / "marl_env.py").exists():
        _s = str(_d)
        # Append (do not prepend): the interpreter already puts ``scripts/`` first so
        # ``import compare_all_rl_convergence`` resolves; appending root enables ``rl_agent``.
        if _s not in sys.path:
            sys.path.append(_s)
        break
else:
    raise RuntimeError(
        "Could not locate repository root (expected rl_agent/marl_env.py). "
        "Run scripts from the UAV-Repositioning repository clone."
    )
