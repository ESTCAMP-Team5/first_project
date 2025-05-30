가설: 평균 기온(AvgTemp)은 커피 매출(amt)에 영향을 미친다
![image](https://github.com/user-attachments/assets/75350a26-6f5b-4018-be34-8cdf86998ef8)
# 기온만으로는 매출을 설명하기 어렵다 -> 다른 요소도 넣어야 한다.
![image](https://github.com/user-attachments/assets/e17b507b-6acc-4e62-85a4-e095f21d7139)
![image](https://github.com/user-attachments/assets/c58000df-ab8d-4bcc-88f0-b6d43d184766)
# 기온은 매출과 거의 상관 없음
![image](https://github.com/user-attachments/assets/63a55e33-7b6e-4d0f-aed7-66a6741904cb)
# 5월, 10월, 12월에 매출이 높음




# 🧪 가설 검증 방법 요약

데이터 전처리: 결측치 제거 및 기온-매출 데이터 정제
회귀 모델: sklearn.linear_model.LinearRegression
통계 검증: scipy.stats.linregress로 p-value 및 결정계수 산출
결과 저장: 예측값과 지표를 포함한 결과를 MySQL의 sales_predictions 테이블에 저장

p-value < 0.05 → 기온은 매출에 통계적으로 유의미한 영향을 미칩니다.
p-value ≥ 0.05 → 기온은 매출에 영향 없음 (귀무가설 채택)
