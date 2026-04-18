"""
move_hard_images.py
===================
Moves the hardest N images (highest cyclic MAE error) to a destination folder.
Edit the config block below, then just run:  python move_hard_images.py
"""

import csv
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
AUDIT_CSV = "audit.csv"       # path to the audit CSV
SRC_DIR   = "dataset"         # source image folder
DST_DIR   = "hard_images"     # destination folder (created if needed)
N         = 100               # number of hardest images to move
COPY      = False             # True = copy, False = move
DRY_RUN   = False             # True = preview only, don't touch files
# ---------------------------------------------------------------------------


def load_audit(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["error_min"] = float(row["error_min"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["error_min"], reverse=True)
    return rows


def main() -> None:
    op = "copy" if COPY else "move"
    dry = " [DRY RUN]" if DRY_RUN else ""

    print(f"\n{'='*60}")
    print(f"  Audit : {AUDIT_CSV}")
    print(f"  Src   : {SRC_DIR}")
    print(f"  Dst   : {DST_DIR}")
    print(f"  N     : {N}  |  op: {op}{dry}")
    print(f"{'='*60}\n")

    rows  = load_audit(AUDIT_CSV)[:N]
    moved = skipped = errors = 0

    if not DRY_RUN:
        os.makedirs(DST_DIR, exist_ok=True)

    for rank, row in enumerate(rows, start=1):
        # Strip the folder prefix baked into the CSV path, use SRC_DIR instead
        p        = Path(row["path"])
        filename = Path(*p.parts[1:]) if len(p.parts) > 1 else p
        src      = Path(SRC_DIR) / filename
        dst      = Path(DST_DIR) / src.name

        # If exact path not found, match by stem ignoring extension
        if not src.exists():
            matches = list(Path(SRC_DIR).glob(f"{src.stem}.*"))
            if matches:
                src = matches[0]
                dst = Path(DST_DIR) / src.name

        tag = f"#{rank:>3} | err={row['error_min']:>7.1f}m | {src.name}"

        if not src.exists():
            print(f"  SKIP  {tag}  (not found)")
            skipped += 1
            continue
        if dst.exists():
            print(f"  SKIP  {tag}  (already in dst)")
            skipped += 1
            continue

        print(f"  {op.upper():4}  {tag}")
        if not DRY_RUN:
            try:
                shutil.copy2(src, dst) if COPY else shutil.move(str(src), str(dst))
                moved += 1
            except Exception as exc:
                print(f"         ERROR: {exc}")
                errors += 1
        else:
            moved += 1

    print(f"\n{'='*60}")
    print(f"  {op.capitalize()}d : {moved}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()