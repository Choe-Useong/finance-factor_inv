from __future__ import annotations

import csv
from pathlib import Path
from typing import List


ENCODING = "cp949"
ROOT = Path("통합")
TYPES = [
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "현금흐름표",
]
SAMPLE_ROWS = 1000


def read_header(path: Path) -> List[str] | None:
    try:
        with open(path, "r", encoding=ENCODING, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            return next(reader, None)
    except Exception:
        return None


def validate_file(path: Path) -> dict:
    result = {
        "file": str(path),
        "header_len": 0,
        "has_item": False,
        "item_idx": -1,
        "has_stmt_kind": False,
        "has_scope": False,
        "period_idx": -1,
        "period_name": None,
        "unnamed_pos": [],
        "period_is_dangi": False,
        "rows_checked": 0,
        "len_mismatches": [],  # list of (row_no, got_len)
    }

    hdr = read_header(path)
    if not hdr:
        return result
    result["header_len"] = len(hdr)
    # 항목명 위치
    try:
        result["item_idx"] = hdr.index("항목명")
        result["has_item"] = True
    except ValueError:
        result["item_idx"] = -1
        result["has_item"] = False

    # 추가 컬럼 존재 여부
    result["has_stmt_kind"] = ("재무제표분류" in hdr)
    result["has_scope"] = ("연결구분" in hdr)

    # 언네임드 컬럼 (빈 문자열로 판단)
    unnamed = []
    for i, col in enumerate(hdr):
        if (col or "").strip() == "":
            unnamed.append(i + 1)  # 1-based
    result["unnamed_pos"] = unnamed

    # 기간 컬럼 확인: '항목명' 이후 첫 '당기*' 컬럼
    if result["has_item"]:
        for j in range(result["item_idx"] + 1, len(hdr)):
            name = (hdr[j] or "").strip()
            if name.startswith("당기"):
                result["period_idx"] = j
                result["period_name"] = name
                result["period_is_dangi"] = (name == "당기")
                break

    # 데이터 행 샘플 길이 검증
    try:
        with open(path, "r", encoding=ENCODING, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)  # skip header
            for i, row in enumerate(reader, start=2):
                result["rows_checked"] += 1
                if len(row) != len(hdr):
                    if len(result["len_mismatches"]) < 5:
                        result["len_mismatches"].append((i, len(row)))
                if result["rows_checked"] >= SAMPLE_ROWS:
                    break
    except Exception:
        pass

    return result


def main() -> int:
    base = ROOT.resolve()
    if not base.exists():
        print("Output root not found:", base)
        return 2

    overall_ok = True
    for t in TYPES:
        dirp = base / t
        if not dirp.exists():
            continue
        print(f"[{t}] {dirp}")
        files = sorted(dirp.glob("*.tsv"), key=lambda p: str(p).lower())
        for fp in files:
            r = validate_file(fp)
            print(f" - {fp.name}: header_len={r['header_len']}, item_idx={r['item_idx']}, period_idx={r['period_idx']}, period='{r['period_name']}'")
            if not r["has_item"]:
                print("   ! '항목명' missing in header")
                overall_ok = False
            if r["unnamed_pos"]:
                print(f"   ! unnamed columns at (1-based): {r['unnamed_pos']}")
                overall_ok = False
            if not r["has_stmt_kind"] or not r["has_scope"]:
                print("   ! missing derived columns: 재무제표분류/연결구분")
                overall_ok = False
            if r["period_idx"] < 0:
                print("   ! cannot find '당기*' column after '항목명'")
                overall_ok = False
            elif not r["period_is_dangi"]:
                print("   ! period column header not normalized to '당기'")
                overall_ok = False
            if r["len_mismatches"]:
                print(f"   ! row length mismatches (first {len(r['len_mismatches'])} shown): {r['len_mismatches']}")
                overall_ok = False
        print("")

    print("OK" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
