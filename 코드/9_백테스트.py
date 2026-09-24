# -*- coding: utf-8 -*-
# 실행 안내: 이 코드는 템플릿입니다. 외부 라이브러리 설치/실행은 사용자 환경에서 직접 하셔야 합니다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ===== 0) 입력 & 파라미터 =====
PRICE_FILE  = r"C:\Users\admin\Desktop\재무제표정리\통합재무\월별첫영업일가격.parquet"
FACTOR_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성_withMOM.parquet"

params = {
    # ---------------------------
    # [선발(랭킹/스코어)에 쓸 팩터]  ex) ["BM"], ["BM","GP_A"]
    # ---------------------------
    "FACTORS": ["BM"],

    # ---------------------------
    # [사전 필터] — 순차 교차 적용(원본 컬럼 기준)
    #   연산자:
    #     ("gt", x)  : > x       ("ge", x): ≥ x
    #     ("lt", x)  : < x       ("le", x): ≤ x
    #     ("between", lo, hi) : lo ≤ x ≤ hi
    #     ("top_pct", p)          : 상위 p%
    #     ("bottom_pct", p)       : 하위 p%
    #     ("top_band_pct", p1, p2): 상위 p1~p2% (예: 0.10~0.50 → 상위 10~50%)
    #     ("rank_band_n", n1, n2) : 상위 n1~n2등(값 큰 순)
    # ---------------------------
    "FACTOR_FILTERS": { "SIZE": ("top_pct",0.80),
        # 예) "BM": ("top_band_pct", 0.10, 0.50),
        # 예) "SIZE": ("rank_band_n", 1, 200),
        # 예) "TURNOVER": ("between", 0.05, 0.30),
    },

    # ---------------------------
    # [표준화 & 윈저화] — 선발용 특징값 생성 제어
    #   STD_METHOD:
    #     "none"     : 표준화 안 함(원본 사용) → 스코어 단계에서 SCORE_NORM 적용
    #     "zscore"   : 연도 스냅 전체 z-score
    #     "rank"     : 연도 스냅 전체 백분위(0~1, 상위가 1에 근접)
    #     "ind_z"    : 연도×업종 내 z-score
    #     "ind_rank" : 연도×업종 내 백분위
    #   DO_WINSOR: True면 표준화 전에 윈저화(분위수 절단)
    #   WINSOR_Q : (하한, 상한) 분위수
    # ---------------------------
    "STD_METHOD": "ind_z",      # "none"|"zscore"|"rank"|"ind_z"|"ind_rank"
    "DO_WINSOR": True,
    "WINSOR_Q": (0.05, 0.95),

    # STD_METHOD="none"일 때만 사용되는 정규화 방식
    "SCORE_NORM": "zscore",       # "rank" | "zscore"

    # 최종 컷 — (필터 후) 상위 비율  (필터만으로 확정하려면 1.0)
    "TOP_PCT": 0.3,

    # 스코어 가중치 (METHOD="score"에서만, 키는 원본 팩터명)
    "FACTOR_WEIGHTS": {
         #"BM": 2, "GP_A": 0.5, "MOM_12_1": 0.5
    },

    # 선발 방법
    "METHOD": "score",          # "intersection" | "score"

    # 업종 컬럼 (업종 표준화/필터에 필요)
    "INDUSTRY_COL": "업종명",

    # 리밸런싱/가중치/벤치마크
    "REB_MONTHS": [7],          #
    "WEIGHT_METHOD": "equal",   # "equal" | "factor"(score 방식에서만 의미)
    "BENCHMARK": "^KS11",

    # 디버그
    "DEBUG": False,             # True → 필터 전/후 로그
}

# ===== 1) 데이터 로드 =====
price  = pd.read_parquet(PRICE_FILE,  engine="pyarrow")
factor = pd.read_parquet(FACTOR_FILE, engine="pyarrow")

price["종목코드"]  = price["종목코드"].astype(str).str.strip().str.zfill(6)
factor["종목코드"] = factor["종목코드"].astype(str).str.strip().str.zfill(6)
price["날짜"]      = pd.to_datetime(price["날짜"], errors="coerce")
factor["결산연도"]  = factor["결산연도"].astype(int)
# 종가 피벗
close = (
    price.pivot(index="날짜", columns="종목코드", values="종가")  # 세로형 가격표를 가로형(날짜×종목코드 매트릭스)로 변환
         .sort_index()                                         # 시계열 연산(shift/rolling)을 위해 날짜 오름차순 정렬
)
close = close.loc[:, ~close.columns.duplicated(keep="first")]  # 혹시 동일 종목코드가 중복 컬럼으로 들어온 경우 첫 번째만 유지하여 중복 제거

# ===== 2) 상폐 처리 =====
proc = close.copy()                # 원본 보존용 복사
last_idx = proc.index[-1]          # 전체 데이터의 마지막 날짜(상폐 판단 기준)
for code in proc.columns:          # 각 종목(컬럼)별로 순회
    s = proc[code]                 # 해당 종목의 월별 종가 시계열(Series)
    lv = s.last_valid_index()      # 마지막 유효 값(결측이 아닌 값)이 있는 날짜
    if lv is None:                 # 전 구간이 전부 NaN이라면(데이터 없음) 건너뛰기
        continue
    # 마지막 유효 시점(lv)이 전체 마지막(last_idx)보다 이전이거나,
    # 마지막 값이 NaN이면 → 그 뒤 구간을 '상폐 이후'로 간주
    if (lv < last_idx) or pd.isna(s.iloc[-1]):
        # lv 이후 구간의 NaN을 1.0으로 채움
        # - 의도: 다음 달 수익률이 크게 음(-)이 되어 포트에서 사실상 탈락(청산)되도록 한 번에 반영
        # - 대안: 마지막 가격으로 고정(수익률 0 유지) 또는 0원 처리(완전 -100%) 등 정책적으로 선택 가능
        proc.loc[lv:, code] = proc.loc[lv:, code].fillna(1.0)

# ===== 유틸: 윈저/표준화/랭크 & 안전 헬퍼 =====
def winsorize_series(s: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    lo = s.quantile(q_low)         # 하단 분위수 컷 기준값 (예: 1%)
    hi = s.quantile(q_high)        # 상단 분위수 컷 기준값 (예: 99%)
    return s.clip(lower=lo, upper=hi)  # 극단치 완화를 위해 구간 밖 값을 경계값으로 잘라냄(윈저화)

def zscore_safe(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)            # 전체 표준편차(ddof=0: 모집단 표준편차). 0이면 분모 1로 대체해 발산 방지
    return (s - s.mean()) / (std if (std and std > 0) else 1.0)

def rank_pct(s: pd.Series) -> pd.Series:
    # pandas pct rank는 ascending=True일 때 큰 값이 1에 가까워진다.
    return s.rank(ascending=True, pct=True)

def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):  # 일부 연산에서 데이터프레임으로 반환될 수 있어 Series로 강제
        s = s.iloc[:, 0]
    return s

# ===== 3) 유니버스 선택 함수 =====
def select_universe(factor_df, year, params):
    # ---- 파라미터 언팩 ----
    factors    = list(params.get("FACTORS", []))          # 선발에 사용할 팩터 목록(컬럼명)
    filters    = params.get("FACTOR_FILTERS", {})         # 사전 필터 규칙 딕셔너리 {컬럼: (연산자, 값[, 값2])}
    top_pct    = float(params.get("TOP_PCT", 0.1))        # (필터 후) 최종 상위 비율
    method     = params.get("METHOD", "score")            # "intersection" or "score"
    score_norm = params.get("SCORE_NORM", "rank")         # STD_METHOD="none"일 때 스코어 단에서 적용할 정규화 방식
    wdict      = params.get("FACTOR_WEIGHTS", {})         # score 방식에서 팩터 가중치
    debug      = bool(params.get("DEBUG", False))         # 디버그 로그 여부

    std_method = params.get("STD_METHOD", "none")         # "none"/"zscore"/"rank"/"ind_z"/"ind_rank"
    do_winsor  = bool(params.get("DO_WINSOR", False))     # 표준화 전에 윈저화 적용 여부
    wq_low, wq_high = params.get("WINSOR_Q", (0.01, 0.99))# 윈저화 분위수 구간

    ind_col    = params.get("INDUSTRY_COL", "업종명")     # 업종명 컬럼(업종 표준화/필터에 필요)

    # --- 스냅: 해당 '결산연도'의 교차섹션 생성 (필요 컬럼만) ---
    filter_cols = list(filters.keys())                    # 필터에 사용되는 컬럼 목록
    # 종목코드/업종 + (팩터 ∪ 필터) 컬럼을 중복 없이 모아 요청
    want_cols = ["종목코드", ind_col] + list(dict.fromkeys([*factors, *filter_cols]))
    # factor_df에 실제 존재하는 컬럼만 취해 과도한 요청(미존재 컬럼) 방지
    have_cols = ["종목코드"] + [c for c in want_cols if (c != "종목코드" and c in factor_df.columns)]

    # 해당 결산연도(year)의 스냅샷 추출
    snap = factor_df.loc[factor_df["결산연도"] == year, have_cols].copy()
    if snap.empty:
        if debug: print(f"[DEBUG {year}] empty snapshot")
        return []

    # 중복 컬럼 제거(merge/concat 과정에서 발생 가능)
    snap = snap.loc[:, ~snap.columns.duplicated(keep="first")]

    # 선발이나 필터에 쓰이는 컬럼에 NaN이 있으면 제거(계산 안정성)
    used_cols = [c for c in set(factors) | set(filter_cols) if c in snap.columns]
    if used_cols:
        snap = snap.dropna(subset=used_cols)
    if snap.empty:
        if debug: print(f"[DEBUG {year}] empty after dropna on used_cols")
        return []

    if debug:
        print(f"[DEBUG {year}] initial universe: {len(snap)}")

    # --- (A) 필터링: '원본' 컬럼 기준으로 순차 적용(AND 연산처럼 교차 축소) ---
    for f, rule in filters.items():
        if f not in snap.columns:
            if debug: print(f"[DEBUG {year}] filter skipped (missing col): {f} {rule}")
            continue

        s = _as_series(snap, f)    # 대상 컬럼(Series)
        op = rule[0]               # 연산자 문자열
        before = len(snap)         # 필터 전 표본 수

        if op == "gt":             # s > x
            snap = snap[s >  rule[1]]
        elif op == "ge":           # s ≥ x
            snap = snap[s >= rule[1]]
        elif op == "lt":           # s < x
            snap = snap[s <  rule[1]]
        elif op == "le":           # s ≤ x
            snap = snap[s <= rule[1]]
        elif op == "between":      # lo ≤ s ≤ hi
            lo, hi = rule[1], rule[2]
            snap = snap[(s >= lo) & (s <= hi)]
        elif op == "top_pct":      # 상위 p%만 남김(값이 클수록 좋다고 가정)
            p = float(rule[1]); k = max(1, int(np.ceil(len(snap) * p)))
            snap = snap.loc[s.nlargest(k).index]
        elif op == "bottom_pct":   # 하위 p%만 남김(역방향 컷)
            p = float(rule[1]); k = max(1, int(np.ceil(len(snap) * p)))
            snap = snap.loc[s.nsmallest(k).index]
        elif op == "top_band_pct": # 상위 p1~p2% 대역만(예: 10~50%)
            p1, p2 = sorted((float(rule[1]), float(rule[2])))
            q_low  = s.quantile(1 - p2)   # 상위 p2%의 하한
            q_high = s.quantile(1 - p1)   # 상위 p1%의 하한(더 큼)
            snap = snap[(s >= q_low) & (s <= q_high)]
        elif op == "rank_band_n":  # 상위 n1~n2등만(값 큰 순)
            n1, n2 = sorted((int(rule[1]), int(rule[2])))
            r = s.rank(ascending=False, method="first")
            snap = snap[(r >= n1) & (r <= n2)]
        # 그 외 연산자는 무시

        after = len(snap)          # 필터 후 표본 수
        if debug: print(f"[DEBUG {year}] filter {f} {rule}: {before} -> {after}")
        if snap.empty:             # 모든 표본이 탈락하면 즉시 종료
            if debug: print(f"[DEBUG {year}] empty after {f} {rule}")
            return []

    # --- (B) 특징값 생성: (선택)윈저화 → 표준화/랭크 (transform 기반으로 인덱스 정합 보장) ---
    sel_factors = []                                   # 선발에 실제 사용할 컬럼명 리스트(원본 or 파생)
    grp_ids = snap[ind_col] if std_method in ("ind_z", "ind_rank") else None  # 업종 표준화 시 그룹 라벨

    for f in factors:
        if f not in snap.columns:                      # 요청한 팩터가 스냅에 없으면 스킵
            continue
        x = _as_series(snap, f).copy()                 # 원본 값 복사(윈저/표준화 전용)

        # (1) 윈저화: 극단치 영향 완화(업종 표준화라면 업종 내에서 분위수 절단)
        if do_winsor:
            if grp_ids is None:
                x = winsorize_series(x, q_low=wq_low, q_high=wq_high)
            else:
                x = x.groupby(grp_ids).transform(lambda s: winsorize_series(s, q_low=wq_low, q_high=wq_high))

        # (2) 표준화/랭크: 전역/업종 내에서 z-score 또는 백분위(상위=좋게)
        if std_method == "none":
            feat = x
            sel_name = f                              # 원본 컬럼 그대로 사용
        elif std_method == "zscore":
            feat = zscore_safe(x)                     # 전역 z-score
            sel_name = f + "_z"
        elif std_method == "rank":
            feat = rank_pct(x)       # 전역 백분위(값 클수록 1에 가까움)
            sel_name = f + "_r"
        elif std_method == "ind_z":
            feat = x.groupby(grp_ids).transform(zscore_safe)              # 업종 내 z-score
            sel_name = f + "_indz"
        elif std_method == "ind_rank":
            feat = x.groupby(grp_ids).transform(lambda s: rank_pct(s))  # 업종 내 백분위
            sel_name = f + "_indr"
        else:
            raise ValueError("Unknown STD_METHOD")

        snap[sel_name] = feat        # transform은 원 인덱스 보존 → 안전하게 대입 가능
        sel_factors.append(sel_name) # 선발에 사용할 최종 컬럼명 등록

    if not sel_factors:              # 사용할 팩터가 하나도 없으면 빈 리스트 반환
        return []

    # --- (C) 최종 선택: 교집합 or 스코어 상위 ---
    if method == "intersection":
        # 각 팩터별로 상위 top_pct를 뽑고, 그 교집합에 속하는 종목만 최종 선택
        selected_sets = []
        for f in sel_factors:
            s = _as_series(snap, f)
            k = max(1, int(np.ceil(len(snap) * top_pct)))
            idx = s.nlargest(k).index
            selected_sets.append(set(snap.loc[idx, "종목코드"]))
        sel = selected_sets[0]
        for ss in selected_sets[1:]:
            sel &= ss                           # 집합 교집합 연산
        return list(sel)

    elif method == "score":
        # 모든 팩터를 정규화 값으로 가중합(FACTORS 순서에 맞춰 wdict의 원본 팩터명 키를 사용)
        sc = pd.Series(0.0, index=snap.index)   # 종목별 총점(초기 0)
        for f_raw, f_sel in zip(factors, sel_factors):
            s = _as_series(snap, f_sel)
            if std_method == "none":
                # 표준화가 없으면 여기서 정규화(rank 또는 zscore)를 적용
                if score_norm == "rank":
                    z = rank_pct(s)
                else:
                    z = zscore_safe(s)
            else:
                z = s  # 이미 전 단계에서 표준화/랭크 끝남
            sc = sc.add(z * float(wdict.get(f_raw, 1.0)), fill_value=0.0)  # 가중합
        snap["score"] = sc
        k = max(1, int(np.ceil(len(snap) * top_pct)))
        return list(snap.nlargest(k, "score")["종목코드"])  # 점수 상위 k개 종목 코드 반환

    else:
        raise ValueError("Unknown METHOD")

# ===== 4) 리밸런싱 날짜 =====
# 월 인덱스에서 파라미터에 지정된 월(REB_MONTHS)에 해당하는 날짜만 추출 → 리밸런싱 시점 리스트
rebal_dates = proc.index[proc.index.month.isin(params["REB_MONTHS"])]

# ===== 5) 가중치 계산 =====
weights = pd.DataFrame(index=proc.index, columns=proc.columns, dtype=float)  # 가중치 타임라인(날짜×종목) 초기화
univ = set(proc.columns)                                                     # 투자 가능 종목 유니버스(컬럼 집합)

for d in rebal_dates:                                   # 각 리밸 날짜별로
    y = d.year                                          # 해당 연도(예: 2025년 7월 리밸 → 재무 스냅은 2024년 결산연도 사용)
    selected = select_universe(factor, y - 1, params)   # 유니버스 선정(결산연도=y-1)
    # 실제 가격 행에 존재하고, 해당 일자의 가격이 NaN이 아닌 종목만 유지(거래 불가/결측 제거)
    selected = [c for c in selected if c in univ and not pd.isna(proc.loc[d, c])]
    if len(selected) == 0:
        continue

    # 리밸런싱 시점에는 기존 보유 종목을 먼저 0으로 초기화한다.
    # 그렇지 않으면 비선정 종목의 과거 비중이 ffill()로 다음 기간까지 남는다.
    weights.loc[d, :] = 0.0

    if params["METHOD"] == "intersection":
        # 교집합 방식에서는 현재 템플릿상 equal-weight만 허용
        if params["WEIGHT_METHOD"] != "equal":
            raise ValueError("교집합 방식에서는 equal-weight만 가능합니다.")
        w = 1.0 / len(selected)                     # 동일가중
        weights.loc[d, selected] = w                # 리밸 날짜에만 가중치 셋(나중에 ffill로 유지)

    elif params["METHOD"] == "score":
        if params["WEIGHT_METHOD"] == "equal":
            w = 1.0 / len(selected)                 # 동일가중
            weights.loc[d, selected] = w
        elif params["WEIGHT_METHOD"] == "factor":
            # 간단 가중치: 선택된 종목들의 '원본 팩터 합'으로 비중 비례
            # (엄밀히 하려면 선발에 쓴 표준화 스킴 그대로 점수 재계산해서 사용 권장)
            snap = factor.loc[factor["결산연도"] == (y - 1), ["종목코드"] + params["FACTORS"]]
            snap = snap.set_index("종목코드").loc[selected]  # 선택 종목만 추림
            scores = snap.sum(axis=1)                        # 여러 팩터를 단순 합(가중 합도 가능)
            scores = scores / scores.sum()                   # 합이 1이 되도록 정규화
            weights.loc[d, selected] = scores                # 리밸 날짜의 개별 종목 가중치 기록
        else:
            raise ValueError("Unknown WEIGHT_METHOD")


# ===== 6) 포트 수익률 =====
rets = proc.pct_change(fill_method=None)
weights_ff = weights.ffill().shift(1)

common_idx = rets.index.intersection(weights_ff.index)
rets = rets.loc[common_idx].fillna(0.0)
weights_ff = weights_ff.loc[common_idx].fillna(0.0)

port_ret = (weights_ff * rets).sum(axis=1)
nav = (1.0 + port_ret).cumprod()
nav.name = "Portfolio"

# ===== 7) 벤치마크 (yfinance) =====
bench = yf.download(params["BENCHMARK"], start='2016-07-01', interval="1mo")
bench = bench[["Close"]].rename(columns={"Close": "종가"})
bench.index.name = "time"

bench_ret = bench["종가"].pct_change()
bench_nav = (1.0 + bench_ret).cumprod()
bench_nav.name = "Benchmark"

# ===== 8) 시각화 (한 플롯만: NAV vs Benchmark) =====
plot_idx = nav.index.intersection(bench_nav.index)
plt.figure(figsize=(10, 4))
ax = nav.loc[plot_idx].plot(label="Portfolio")
bench_nav.loc[plot_idx].plot(label="Benchmark", ax=ax)
plt.legend()
plt.title("Portfolio vs Benchmark")
plt.tight_layout()
plt.show()

# ===== 9) 결과 출력 (텍스트만) =====
print("[INFO] 기간:", nav.index.min(), "→", nav.index.max())
print("[INFO] 리밸 횟수:", int(weights.notna().any(axis=1).sum()))
if not nav.dropna().empty:
    print("[INFO] 마지막 NAV:", round(float(nav.dropna().iloc[-1]), 6))
if not bench_nav.dropna().empty:
    print("[INFO] 마지막 벤치마크 NAV:", round(float(bench_nav.dropna().iloc[-1]), 6))

print("\n[월수익률 head]\n", port_ret.dropna().head())
print("\n[NAV head]\n", nav.dropna().head())
print("\n[벤치마크 NAV head]\n", bench_nav.dropna().head())

# ===== 10) 리밸 시점별 선택 종목 수 (텍스트만) =====
selected_counts = weights.loc[rebal_dates].notna().sum(axis=1)
print("\n[리밸런싱 시점별 종목 수]\n", selected_counts)
