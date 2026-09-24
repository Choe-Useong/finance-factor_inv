import pandas as pd
import re

# -------------------------------
# 1) 데이터 로드 및 기본 필터링
# -------------------------------
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"
df = pd.read_parquet(FILE_PATH, engine="pyarrow")

# 비금융 & 연결 재무제표만 사용
exclude_industries = ["은행 및 저축기관", "금융 지원 서비스업", "보험업"]
df_non_fin = df[
    (~df["업종명"].isin(exclude_industries)) &
    (df["연결구분"] == "연결")
]

# -------------------------------
# 2) IFRS 지배/비지배 태그 모두 없는 케이스
# -------------------------------
grouped = (
    df_non_fin.groupby(["종목코드", "결산기준일"])["항목코드_통일"]
    .apply(set)
    .reset_index(name="항목세트")
)

def both_missing(items):
    return (
        "EquityAttributableToOwnersOfParent" not in items and
        "NoncontrollingInterests" not in items
    )

cases_both_missing = grouped[grouped["항목세트"].apply(both_missing)]

# -------------------------------
# 3) 원본과 조인 후 계정명 전처리
# -------------------------------
merged = df_non_fin.merge(
    cases_both_missing[["종목코드","결산기준일"]],
    on=["종목코드","결산기준일"],
    how="inner"
)

def clean_label(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"\s+", "", x)          # 공백 제거
    x = re.sub(r"[^가-힣0-9]", "", x)  # 한글/숫자 외 제거
    return x

merged["항목명_clean"] = merged["항목명"].apply(clean_label)

# -------------------------------
# 4) 지배주주지분 라벨 후보 정의
# -------------------------------
owner_labels = [
    "지배기업소유주지분","지배기업의소유주지분","지배기업의소유주에게귀속되는자본",
    "지배기업의소유지분","지배회사소유지분","지배기업소유지분",
    "지배기업소유주에귀속되는자본","지배주주지분","지배기업소유지분합계",
    "지배기업지분","지배기업의소유주에게귀속되는지분","지배기업주주지분"
]

grouped2 = (
    merged.groupby(["종목코드","결산기준일"])["항목명_clean"]
    .apply(set)
    .reset_index(name="항목세트_clean")
)

# -------------------------------
# 5) 조건별 케이스 분류
# -------------------------------
# (1) 지배 후보 라벨 없는 케이스
cases_no_owner_labels = grouped2[
    grouped2["항목세트_clean"].apply(lambda items: all(lbl not in items for lbl in owner_labels))
]

# (2) 지배 후보 라벨 없음 + 비지배 키워드 없음
cases_no_equity_clues = grouped2[
    grouped2["항목세트_clean"].apply(
        lambda items: all(lbl not in items for lbl in owner_labels) and
                      all(("비지배" not in itm and "미지배" not in itm) for itm in items)
    )
]

# (3) Equity 자체도 없는 케이스 (항목코드 기준)
cases_no_equity_final = cases_both_missing[
    cases_both_missing["항목세트"].apply(lambda items: "Equity" not in items)
]

# -------------------------------
# 6) 결과 출력
# -------------------------------
print("IFRS 지배/비지배 모두 없는 케이스 수:", len(cases_both_missing))
print("지배 후보 라벨도 없는 케이스 수:", len(cases_no_owner_labels))
print("지배/비지배 라벨 모두 없는 케이스 수:", len(cases_no_equity_clues))
print("Equity 마저 없는 최종 케이스 수:", len(cases_no_equity_final))



# (3) Equity 자체도 없는 케이스 (항목코드 기준, 전체 기준)
cases_no_equity_final = grouped[
    grouped["항목세트"].apply(
        lambda items: (
            "EquityAttributableToOwnersOfParent" not in items and
            "NoncontrollingInterests" not in items and
            "Equity" not in items
        )
    )
]

print("Equity까지 전부 없는 최종 케이스 수:", len(cases_no_equity_final))
print(cases_no_equity_final.head(20))




# (4) 자산(Assets) + 부채와자본총계(EquityAndLiabilities) 모두 없는 케이스
cases_no_assets_final = grouped[
    grouped["항목세트"].apply(
        lambda items: ("Assets" not in items) and ("EquityAndLiabilities" not in items)
    )
]

print("자산(Assets)과 부채+자본총계(EquityAndLiabilities) 모두 없는 케이스 수:", len(cases_no_assets_final))
print(cases_no_assets_final.head(20))



# (5) 자산/부채+자본 총계 모두 없고, 항목명에도 '자산총계' 라벨이 없는 케이스
# 우선 후보 케이스만 원본과 조인
merged_assets = df_non_fin.merge(
    cases_no_assets_final[["종목코드","결산기준일"]],
    on=["종목코드","결산기준일"],
    how="inner"
)

# 항목명 정리
def clean_label(x: str) -> str:
    if pd.isna(x): return x
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^가-힣0-9]", "", x)
    return x

merged_assets["항목명_clean"] = merged_assets["항목명"].apply(clean_label)

# 그룹별로 '자산총계' 포함 여부 확인
grouped_assets = (
    merged_assets.groupby(["종목코드","결산기준일"])["항목명_clean"]
    .apply(set)
    .reset_index(name="항목세트_clean")
)

def no_assets_label(items):
    return all("자산총계" not in itm for itm in items)

cases_no_assets_label = grouped_assets[grouped_assets["항목세트_clean"].apply(no_assets_label)]

print("자산/부채+자본 총계 없고, 항목명에도 '자산총계' 없는 케이스 수:", len(cases_no_assets_label))
print(cases_no_assets_label.head(20))

