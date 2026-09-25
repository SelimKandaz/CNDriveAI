from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cndriveai.dataset import heldout_records, train_records, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    train = train_records()
    heldout = heldout_records()
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "heldout.jsonl", heldout)
    print(f"synthetic_train={len(train)} heldout={len(heldout)} out={args.out}")


if __name__ == "__main__":
    main()
