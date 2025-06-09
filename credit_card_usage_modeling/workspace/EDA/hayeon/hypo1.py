import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1️⃣ MySQL 연결 정보 설정
DB_USER = "root"
DB_PASSWORD = "1234"
DB_HOST = "localhost"       
DB_PORT = "3306"            
DB_NAME = "suwon"           

# 2️⃣ SQLAlchemy 연결 엔진 생성
@st.cache_resource
def get_engine():
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)
    return engine

# 3️⃣ 데이터 불러오기
@st.cache_data
def load_data():
    query = "SELECT * FROM sales_predictions"
    df = pd.read_sql(query, con=get_engine())
    return df

# 4️⃣ Streamlit UI 시작
st.set_page_config(page_title="기온 vs 커피 매출 분석", layout="wide")
st.title("☕ 기온과 커피 매출 간 상관관계 분석")

df = load_data()

st.markdown("""
    <style>
    /* 전체 배경색 */
    .main {
        background-color: #f9f9f9;
    }
    /* 제목 스타일 */
    h1, h2, h3 {
        font-family: 'Pretendard', sans-serif;
        color: #3B3B3B;
    }
    /* 카드형 컨테이너 */
    .stMarkdown {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    /* 버튼 색상 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# transaction_date 컬럼이 있으면 datetime 변환 후 date 컬럼으로 복사
if 'transaction_date' in df.columns:
    df['date'] = pd.to_datetime(df['transaction_date'])
else:
    st.warning("⚠️ 'transaction_date' 컬럼이 없습니다. 데이터 컬럼명을 확인해주세요.")

# 📋 데이터 미리보기
st.subheader("📊 예측 데이터 미리보기")
st.dataframe(df.head())

# 📈 시각화: 실제 vs 예측
st.subheader("📉기온 변화에 따른 커피 매출: 실제 매출과 예측 비교")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="avg_temp", y="transaction_amount", label="실제 매출", ax=ax)
sns.lineplot(data=df.sort_values("avg_temp"), x="avg_temp", y="transaction_amount", label="예측 매출", color="orange", ax=ax)
ax.set_xlabel("평균 기온 (°C)")
ax.set_ylabel("커피 매출")
ax.set_title("기온에 따른 커피 매출 분석")
st.pyplot(fig)

# ➕ 회귀분석 그래프 추가
st.subheader("📈기온이 커피 매출에 미치는 영향 분석")
fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
sns.regplot(data=df, x="avg_temp", y="transaction_amount", scatter_kws={'s':20, 'alpha':0.6}, line_kws={"color": "red"}, ax=ax_reg)
ax_reg.set_xlabel("평균 기온 (°C)")
ax_reg.set_ylabel("커피 매출")
ax_reg.set_title("기온과 커피 매출 간 회귀선")
st.pyplot(fig_reg)

# 📈 시계열 분석: 날짜별 평균 기온과 평균 매출 추이
if 'date' in df.columns:
    st.subheader("📅날짜별 기온 변화와 커피 매출 흐름")
    df_ts = df.groupby('date').agg({'avg_temp':'mean', 'transaction_amount':'mean'}).reset_index()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df_ts['date'], df_ts['avg_temp'], label='평균 기온 (°C)', color='tab:blue')
    ax2.set_ylabel('평균 기온 (°C)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax3 = ax2.twinx()
    ax3.plot(df_ts['date'], df_ts['transaction_amount'], label='평균 커피 매출', color='tab:orange')
    ax3.set_ylabel('평균 커피 매출', color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')

    ax2.set_xlabel("날짜")
    ax2.set_title("날짜별 평균 기온과 평균 커피 매출 추이")
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    # 📊 히트맵: 월별-요일별 평균 매출 (예시)
    st.subheader("📊월별·요일별 커피 매출 패턴 분석")
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()  # 한글로 하려면 .dt.day_name(locale='ko_KR') 가능
    pivot_table = df.pivot_table(values='transaction_amount', index='weekday', columns='month', aggfunc='mean')

    # 요일 순서 지정 (월~일)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(weekday_order)

    fig3, ax4 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax4)
    ax4.set_title("월별-요일별 평균 커피 매출 히트맵")
    ax4.set_xlabel("월")
    ax4.set_ylabel("요일")
    st.pyplot(fig3)
else:
    st.warning("⚠️ 'date' 컬럼이 없어 시계열 분석 및 히트맵을 그릴 수 없습니다.")

# 📊 회귀 평가 지표
if 'p_value' in df.columns and 'r_squared' in df.columns:
    st.subheader("📐 회귀 분석 지표")
    p_value = df['p_value'].iloc[0]
    r_squared = df['r_squared'].iloc[0]
    st.write(f"**R² (결정계수):** {r_squared:.4f}")
    st.write(f"**P-value:** {p_value:.4f}")
    if p_value < 0.05:
        st.success("✅ 기온은 매출에 통계적으로 유의미한 영향을 미칩니다.")
    else:
        st.warning("❌ 기온은 매출에 유의미한 영향을 미치지 않습니다.")

# 📁 예측 데이터 상세 보기 옵션
if st.checkbox("🔍 예측 결과 상세 보기"):
    st.dataframe(df[['date', 'avg_temp', 'transaction_amount']])
# 📊 데이터 다운로드 기능
if st.button("📥 예측 데이터 다운로드"):
    csv = df.to_csv(index=False)
    st.download_button(label="CSV로 다운로드", data=csv, file_name='sales_predictions.csv', mime='text/csv')
