가설: 평균 기온(AvgTemp)은 커피 매출(amt)에 영향을 미친다

🖥️ Streamlit 데모 화면
[<img src="demo_screenshot.png" width="600"/>](http://localhost:8501/)

🧪 가설 검증 방법 요약
데이터 전처리: 결측치 제거 및 기온-매출 데이터 정제
회귀 모델: sklearn.linear_model.LinearRegression
통계 검증: scipy.stats.linregress로 p-value 및 결정계수 산출
결과 저장: 예측값과 지표를 포함한 결과를 MySQL의 sales_predictions 테이블에 저장

p-value < 0.05 → 기온은 매출에 통계적으로 유의미한 영향을 미칩니다.
p-value ≥ 0.05 → 기온은 매출에 영향 없음 (귀무가설 채택)
