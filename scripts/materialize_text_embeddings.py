from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.config import load_training_config
from puppy.training.text_embedding import materialize_text_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen semantic Text Stream embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = materialize_text_embeddings(
        load_training_config(args.config),
        local_files_only=args.local_files_only,
        force=args.force,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
