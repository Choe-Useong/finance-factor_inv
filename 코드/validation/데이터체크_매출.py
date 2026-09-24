import pandas as pd

# 1) 데이터 로드
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"
df = pd.read_parquet(FILE_PATH, engine="pyarrow")

# 2) 비금융만 필터
exclude_industries = ["은행 및 저축기관", "금융 지원 서비스업", "보험업"]
df_non_fin = df[~df["업종명"].isin(exclude_industries)].copy()






# 회사+기간별 제표종류 집계
check = (
    df_non_fin.groupby(["종목코드","결산기준일"])["제표종류앞"]
    .apply(set)
    .reset_index(name="제표세트")
)

# 손익계산서 없는 케이스
no_is = check[~check["제표세트"].apply(lambda s: any("손익" in x for x in s))]
print("손익계산서 자체가 없는 케이스 수:", len(no_is))
print(no_is.head(20))







# 3) 손익계산서만
df_non_fin = df_non_fin[df_non_fin["제표종류앞"] == "손익계산서"]



# IFRS 태그 정의
REVENUE_TAG = "Revenue"
COGS_TAG    = "CostOfSales"
GP_TAG      = "GrossProfit"

# 회사+결산일별 항목세트
grouped_rev = (
    df_non_fin.groupby(["종목코드","결산기준일"])["항목코드_통일"]
    .apply(set)
    .reset_index(name="항목세트")
)

# (1) 매출총이익 없는 케이스
cases_no_gp = grouped_rev[
    grouped_rev["항목세트"].apply(lambda s: GP_TAG not in s)
]

print("매출총이익 없는 케이스 수:", len(cases_no_gp))
print(cases_no_gp.head(20))

# (2) 그중에서 매출액 & 매출원가 둘 다 없는 케이스
cases_no_gp_rev_cogs = cases_no_gp[
    cases_no_gp["항목세트"].apply(lambda s: (REVENUE_TAG not in s) and (COGS_TAG not in s))
]

print("\n매출총이익 없고, 매출액 & 매출원가도 모두 없는 케이스 수:", len(cases_no_gp_rev_cogs))
print(cases_no_gp_rev_cogs.head(20))


# (3) 매출총이익 없는 케이스 원본과 조인
merged_no_gp_rev_cogs = df_non_fin.merge(
    cases_no_gp_rev_cogs[["종목코드","결산기준일"]],
    on=["종목코드","결산기준일"],
    how="inner"
)

# 항목명 클린업
import re
def clean_label(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^가-힣0-9]", "", x)
    return x

merged_no_gp_rev_cogs["항목명_clean"] = merged_no_gp_rev_cogs["항목명"].apply(clean_label)

# 회사별 항목명 세트
grouped_labels = (
    merged_no_gp_rev_cogs.groupby(["종목코드","결산기준일"])["항목명_clean"]
    .apply(set)
    .reset_index(name="항목명세트")
)

# '매출총이익' 라벨 전혀 없는 케이스
cases_no_gp_label = grouped_labels[
    grouped_labels["항목명세트"].apply(lambda s: all("매출총이익" not in itm for itm in s))
]

print("매출총이익 태그 없고, Revenue/COGS도 없고, 항목명에 '매출총이익' 라벨조차 없는 케이스 수:", len(cases_no_gp_label))
print(cases_no_gp_label.head(20))




# (4) 매출총이익 없음 + Revenue/CostOfSales 없음 케이스 원본과 조인
merged_no_gp_rev_cogs = df_non_fin.merge(
    cases_no_gp_rev_cogs[["종목코드","결산기준일"]],
    on=["종목코드","결산기준일"],
    how="inner"
)

# 항목명 클린업 (앞에서 썼던 함수 재사용)
def clean_label(x: str) -> str:
    import re
    if pd.isna(x):
        return x
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^가-힣0-9]", "", x)
    return x

merged_no_gp_rev_cogs["항목명_clean"] = merged_no_gp_rev_cogs["항목명"].apply(clean_label)

# 회사별 항목명 세트
grouped_labels2 = (
    merged_no_gp_rev_cogs.groupby(["종목코드","결산기준일"])["항목명_clean"]
    .apply(set)
    .reset_index(name="항목명세트")
)

# '매출액'과 '매출원가' 라벨이 전혀 없는 케이스만 필터
cases_no_sales_labels = grouped_labels2[
    grouped_labels2["항목명세트"].apply(
        lambda s: all("매출액" not in itm for itm in s) and
                  all("매출원가" not in itm for itm in s)
    )
]

print("매출총이익 없음 + Revenue/COGS 없음 + 항목명에도 '매출액/매출원가' 없는 케이스 수:", len(cases_no_sales_labels))
print(cases_no_sales_labels.head(20))

# (이전 단계 결과 cases_no_sales_labels 활용)

# 원본과 조인
merged_no_sales = df_non_fin.merge(
    cases_no_sales_labels[["종목코드","결산기준일"]],
    on=["종목코드","결산기준일"],
    how="inner"
)

# 항목명 전처리
import re
def clean_label(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"\s+", "", x)         # 공백 제거
    x = re.sub(r"[^가-힣0-9]", "", x) # 한글/숫자 외 제거
    return x

merged_no_sales["항목명_clean"] = merged_no_sales["항목명"].apply(clean_label)

# 회사별 항목명 세트
grouped_labels_final = (
    merged_no_sales.groupby(["종목코드","결산기준일"])["항목명_clean"]
    .apply(set)
    .reset_index(name="항목명세트")
)

# '영업수익'과 '영업비용' 라벨이 전혀 없는 케이스만 추가 필터
cases_no_sales_no_oper = grouped_labels_final[
    grouped_labels_final["항목명세트"].apply(
        lambda s: all("영업수익" not in itm for itm in s) and
                  all("영업비용" not in itm for itm in s)
    )
]

print("매출총이익 없음 + Revenue/COGS 없음 + 항목명에 '매출액/매출원가' 없음 + '영업수익/영업비용'도 없는 케이스 수:", len(cases_no_sales_no_oper))
print(cases_no_sales_no_oper.head(20))










