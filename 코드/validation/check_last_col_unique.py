from __future__ import annotations

import sys
from pathlib import Path

ENCODING = "cp949"
ROOT = Path("통합")
TYPES = [
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "현금흐름표",
]


def read_header(path: Path) -> list[str] | None:
    try:
        with open(path, "r", encoding=ENCODING, errors="replace") as f:
            line = f.readline().rstrip("\r\n")
    except Exception:
        return None
    if not line:
        return None
    return [t.strip() for t in line.split("\t")]


def main(argv: list[str]) -> int:
    base = ROOT.resolve()
    if not base.exists():
        print("Output root not found:", base)
        return 2

    overall_ok = True
    for t in TYPES:
        d = base / t
        if not d.exists():
            continue
        files = sorted(d.glob("*.tsv"), key=lambda p: str(p).lower())
        period_map: dict[str, list[Path]] = {}
        for f in files:
            hdr = read_header(f)
            if not hdr:
                continue
            # find first period column after '항목명'
            name = ""
            try:
                idx_item = hdr.index("항목명")
            except ValueError:
                idx_item = -1
            if idx_item >= 0:
                for j in range(idx_item + 1, len(hdr)):
                    col = (hdr[j] or "").strip()
                    if col.startswith("당기"):
                        name = col
                        break
            period_map.setdefault(name, []).append(f)
        uniq = list(period_map.keys())
        print(f"[{t}] files={len(files)} unique_period_col={len(uniq)}")
        for name in uniq:
            lab = name or '(missing)'
            print(f" - period='{lab}' -> {len(period_map[name])} files")
        if len(uniq) != 1 or uniq[0] != '당기':
            overall_ok = False
            # show offending files per group
            for name, flist in period_map.items():
                if name == '당기':
                    continue
                for fp in flist[:5]:
                    print(f"   * {fp}")
        print("")

    print("OK" if overall_ok else "MISMATCH FOUND")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
