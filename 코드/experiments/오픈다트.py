import os
import OpenDartReader
import pandas as pd

# 1) API 연결
api_key = os.environ["DART_API_KEY"]
dart = OpenDartReader(api_key)

# 2) 재무제표 데이터 가져오기
# reprt_code='11011' → 사업보고서(연간)
df = dart.finstate('024800', 2022, reprt_code='11011')

# 3) DataFrame으로 변환 (이미 DataFrame일 수 있음)
df = pd.DataFrame(df)

# 4) 엑셀 저장
output_path = "024800_2023_사업보고서.xlsx"
df.to_excel(output_path, index=False, engine="openpyxl")

print(f"저장 완료: {output_path}")
