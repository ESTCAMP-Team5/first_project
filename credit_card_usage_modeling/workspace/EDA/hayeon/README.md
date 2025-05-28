# 가설 설정
어떤 시기(시간대, 요일, 월)에 소비가 증가하는가? = 마케팅 타이밍 포착

지역별로 소비가 집중되는 업종은 무엇인가? = 예산을 어디에 쓸지 결정

소비량이 지속적으로 증가하는 지역/업종은 어디인가? = 성장 가능 업종 파악

소비 변화량이 급증한 시기에는 어떤 외부 요인이 있었는가? = 마케팅 캠페인의 효과 가능성 추론

고소비 지역의 소비 특성과 저소비 지역의 차이는 무엇인가? = 타겟 구분 기준 마련하기 (예: 유동인구, 주거지 비율 등)


#EDA 분석용

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 폰트 깨짐 방지 (한글용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. CSV 불러오기
df = pd.read_csv("2025_gyeonggi_card.csv")

# 2. 날짜 관련 파싱
df['ta_ymd'] = pd.to_datetime(df['ta_ymd'], format="%Y%m%d")
df['year'] = df['ta_ymd'].dt.year
df['month'] = df['ta_ymd'].dt.month
df['dayofweek'] = df['ta_ymd'].dt.dayofweek  # 0=월 ~ 6=일

# 🧪 가설 1: 시기(시간, 요일, 월)에 따른 소비량 분석
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

sns.barplot(x='hour', y='amt', data=df, estimator='sum', ci=None, ax=axes[0])
axes[0].set_title("시간대별 총 소비금액")

sns.barplot(x='dayofweek', y='amt', data=df, estimator='sum', ci=None, ax=axes[1])
axes[1].set_title("요일별 총 소비금액")
axes[1].set_xticklabels(['월', '화', '수', '목', '금', '토', '일'])

sns.barplot(x='month', y='amt', data=df, estimator='sum', ci=None, ax=axes[2])
axes[2].set_title("월별 총 소비금액")

plt.tight_layout()
plt.show()



#월별 소비 추세 시각화

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
monthly = df.groupby(['region', 'industry', 'month'])['amount'].sum().reset_index()

sns.lineplot(data=monthly, x='month', y='amount', hue='region')
plt.title("월별 지역별 소비 추세")
plt.show()

#업종별 소비 집중 지역 파악

pivot = df.pivot_table(index='industry', columns='region', values='amount', aggfunc='sum')
sns.heatmap(pivot, cmap='Blues', annot=True, fmt='.0f')
plt.title("업종별 지역 소비 집중도")
plt.show()

#소비 증가율이 높은 업종 찾기

df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
monthly = df.groupby(['industry', 'month'])['amount'].sum().reset_index()
monthly['growth'] = monthly.groupby('industry')['amount'].pct_change()

top_growth = monthly.sort_values('growth', ascending=False).head(10)
print(top_growth)

#소비당 건수 / 방문자 수

df['amount_per_visit'] = df['amount'] / df['count']
sns.boxplot(data=df, x='region', y='amount_per_visit', hue='industry')
plt.title("방문 1회당 소비금액 분포")
plt.show()
