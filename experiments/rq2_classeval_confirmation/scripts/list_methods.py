#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--activation")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    methods = list(config["phase_orders"][args.phase])
    if args.phase == "q224_conditional":
        if not args.activation:
            raise ValueError("q224_conditional requires --activation")
        activation = json.loads(Path(args.activation).read_text(encoding="utf-8"))
        tasks = set(activation["tasks"])
        methods = [name for name in methods if name.split("_", 1)[0] in tasks]
    print("\n".join(methods))


if __name__ == "__main__":
    main()
