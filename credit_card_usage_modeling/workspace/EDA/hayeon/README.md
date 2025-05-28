# 가설 설정

어떤 시기(시간대, 요일, 월)에 소비가 증가하는가? = 마케팅 타이밍 포착

지역별(수원 용인 화성 안산)로 소비가 집중되는 업종은 무엇인가? = 예산을 어디에 쓸지 결정

소비량이 지속적으로 증가하는 지역/업종은 어디인가? = 성장 가능 업종 파악

소비 변화량이 급증한 시기에는 어떤 외부 요인이 있었는가? = 마케팅 캠페인의 효과 가능성 추론

고소비 지역의 소비 특성과 저소비 지역의 차이는 무엇인가? = 타겟 구분 기준 마련하기

# 파이썬 코드

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

# 📊 분석 그래프 이미지
1. 📈 시간대별 소비 금액
   
![image](https://github.com/user-attachments/assets/500049f2-ab0f-4526-9ae5-e4c08902602a)

2. 📅 요일별 소비 금액
![image](https://github.com/user-attachments/assets/de57724b-1471-46db-a540-6cf51e36ca99)

3. 📆 월별 소비 금액
   ![image](https://github.com/user-attachments/assets/b9e959e4-20aa-4b42-b588-5bf92eecee9c)



# 🧪 가설 2: 지역별 소비가 집중된 업종
target_regions = ['41110', '41460', '41590', '41270']  # 수원, 용인, 화성, 안산
filtered_df = df[df['cty_rgn_no'].isin(target_regions)]

# 시각화 스타일
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="Set2")

# ✅ 1. 지역별 총 소비금액 비교
plt.figure(figsize=(7, 5))
sns.barplot(x='cty_rgn_no', y='amt', data=filtered_df, estimator=np.sum, ci=None)
plt.title("4개 지역별 총 소비금액 비교", fontsize=14)
plt.xlabel("시군 코드", fontsize=12)
plt.ylabel("총 소비금액", fontsize=12)
plt.tight_layout()
plt.savefig("4개지역_총소비금액.png")
plt.show()

# ✅ 2. 지역별 업종별 소비금액 비교
region_buz = filtered_df.groupby(['cty_rgn_no', 'card_tpbuz_nm_1'])['amt'].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='cty_rgn_no', y='amt', hue='card_tpbuz_nm_1', data=region_buz, estimator=np.sum, ci=None)
plt.title("4개 지역별 업종별 소비금액", fontsize=14)
plt.xlabel("시군 코드", fontsize=12)
plt.ylabel("총 소비금액", fontsize=12)
plt.legend(title="업종", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("4개지역_업종소비비교.png")
plt.show()


# 🧪 가설 3: 소비량이 증가하는 지역/업종
monthly_trend = df.groupby(['month', 'cty_rgn_no', 'card_tpbuz_nm_1'])['amt'].sum().reset_index()
monthly_pivot = monthly_trend.pivot_table(index=['cty_rgn_no', 'card_tpbuz_nm_1'], columns='month', values='amt').fillna(0)
monthly_pivot['증가율'] = (monthly_pivot[3] - monthly_pivot[1]) / monthly_pivot[1].replace(0, 1)

print("\n[소비 증가율 상위 지역/업종]")
print(monthly_pivot.sort_values('증가율', ascending=False).head(10))

# 🧪 가설 4: 소비 급증 시기 → 외부 요인 탐지 (시각화용)
daily_amt = df.groupby('ta_ymd')['amt'].sum()
daily_amt.plot(title="일별 소비 추이 (이상치 확인용)", figsize=(12, 4))
plt.ylabel("총 소비 금액")
plt.xlabel("날짜")
plt.grid()
plt.show()

# 🧪 가설 5: 고소비 vs 저소비 지역 비교
region_total = df.groupby('cty_rgn_no')['amt'].sum().sort_values(ascending=False)
top_region = region_total.head(3).index
bottom_region = region_total.tail(3).index

print("\n[고소비 지역 상위 3]")
print(region_total.head(3))
print("\n[저소비 지역 하위 3]")
print(region_total.tail(3))

# 고소비 vs 저소비 지역의 업종 분포 비교
for region in [*top_region, *bottom_region]:
    temp = df[df['cty_rgn_no'] == region]
    temp_group = temp.groupby('card_tpbuz_nm_1')['amt'].sum().sort_values(ascending=False).head(5)
    temp_group.plot(kind='bar', title=f"업종별 소비 상위 (지역: {region})", figsize=(6, 3))
    plt.ylabel("총 소비금액")
    plt.tight_layout()
    plt.show()







