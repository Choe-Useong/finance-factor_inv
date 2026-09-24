from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ENCODING = "cp949"
INPUT_ROOT = Path("통합")
DEFAULT_TYPES = [
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "현금흐름표",
]


def norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def collect_code_names(dir_path: Path) -> Dict[str, Counter]:
    """Scan all TSVs under dir_path and collect name frequencies per code.

    Returns: dict[code] -> Counter({name: count})
    """
    code2names: Dict[str, Counter] = defaultdict(Counter)
    files = sorted(dir_path.glob("*.tsv"), key=lambda p: str(p).lower())
    for fp in files:
        with open(fp, "r", encoding=ENCODING, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                continue
            # find required columns
            try:
                idx_code = header.index("항목코드")
                idx_name = header.index("항목명")
            except ValueError:
                continue
            for row in reader:
                if len(row) <= idx_name:
                    continue
                code = norm_text(row[idx_code])
                name = norm_text(row[idx_name])
                if not code:
                    continue
                code2names[code][name] += 1
    return code2names


def top2(counter: Counter) -> Tuple[str, str]:
    if not counter:
        return "", ""
    mc = counter.most_common(2)
    first = mc[0][0] if len(mc) >= 1 else ""
    second = mc[1][0] if len(mc) >= 2 else ""
    return first, second


def write_excel_or_csv_multi(cat_rows: Dict[str, List[Tuple[str, str, str]]], out_path_base: Path) -> List[Path]:
    """Write per-category sheets (ifrs/dart/other). Returns list of written paths."""
    paths: List[Path] = []
    try:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        first = True
        for cat in ("ifrs", "dart", "other"):
            rows = cat_rows.get(cat, [])
            if first:
                ws = wb.active
                ws.title = cat
                first = False
            else:
                ws = wb.create_sheet(title=cat)
            ws.append(["항목코드", "최빈항목명", "2순위 항목명"])
            for code, n1, n2 in rows:
                ws.append([code, n1, n2])
        out_xlsx = out_path_base.with_suffix(".xlsx")
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        wb.save(out_xlsx)
        paths.append(out_xlsx)
        return paths
    except Exception:
        # Fallback: write separate CSVs
        out_path_base.parent.mkdir(parents=True, exist_ok=True)
        for cat in ("ifrs", "dart", "other"):
            rows = cat_rows.get(cat, [])
            out_csv = out_path_base.with_name(out_path_base.name + f"_{cat}").with_suffix(".csv")
            with open(out_csv, "w", encoding=ENCODING, newline="") as f:
                w = csv.writer(f)
                w.writerow(["항목코드", "최빈항목명", "2순위 항목명"])
                for code, n1, n2 in rows:
                    w.writerow([code, n1, n2])
            paths.append(out_csv)
        return paths


def build_for_type(fin_type: str, input_root: Path, out_dir: Path) -> List[Path] | None:
    src_dir = input_root / fin_type
    if not src_dir.exists():
        print(f"Skip {fin_type}: not found {src_dir}")
        return None
    code2names = collect_code_names(src_dir)
    # categorize
    cats: Dict[str, List[Tuple[str, str, str]]] = {"ifrs": [], "dart": [], "other": []}
    for code in sorted(code2names.keys()):
        n1, n2 = top2(code2names[code])
        if code.startswith("ifrs_"):
            cats["ifrs"].append((code, n1, n2))
        elif code.startswith("dart_"):
            cats["dart"].append((code, n1, n2))
        else:
            cats["other"].append((code, n1, n2))
    out_base = out_dir / f"{fin_type}_codes"
    written = write_excel_or_csv_multi(cats, out_base)
    for p in written:
        print(f"Wrote: {p}")
    return written


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build per-statement code catalogs (top-2 names)")
    ap.add_argument("--types", nargs="*", default=DEFAULT_TYPES, help="Statement types under 통합/ (default: all)")
    ap.add_argument("--in-root", default=str(INPUT_ROOT), help="Input root (default: 통합)")
    ap.add_argument("--out-dir", default=str(Path("코드") / "catalogs"), help="Output directory")
    args = ap.parse_args(argv[1:])

    in_root = Path(args.in_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    written = 0
    for t in args.types:
        if build_for_type(t, in_root, out_dir):
            written += 1
    print(f"Done. Files written: {written}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
