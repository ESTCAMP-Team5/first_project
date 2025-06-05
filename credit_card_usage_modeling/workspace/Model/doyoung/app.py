# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# 로컬 모듈 import
from data_repository import create_data_repository
from models import CustomerPurchaseAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="수원시 구별 매출 예측",
    page_icon="☕",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """데이터 로드"""
    repo = create_data_repository()
    return repo.get_customer_data()


@st.cache_resource
def load_analyzer():
    """분석기 로드"""
    return CustomerPurchaseAnalyzer()


def main():
    # 헤더
    st.markdown('<h1 class="main-title">수원시 구별 커피 매출 예측</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle"구 매출 예측 시스템</p>', unsafe_allow_html=True)

    # 데이터 로드 및 모델 학습
    try:
        with st.spinner("데이터 로드 및 모델 학습 중..."):
            data = load_data()
            analyzer = load_analyzer()

            # 모델 학습 (한 번만 실행)
            if not analyzer.is_fitted:
                analyzer.fit_model(data)

    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {str(e)}")
        return

    st.markdown("---")
    st.header("수원시 4개 구별 매출 예측")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 기본 정보
        gender = st.selectbox("성별", ["여성", "남성"])

        # 예측 버튼

    with col2:
        age_group = st.selectbox("연령대", [1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11], format_func=lambda x: f"연령대 {x}")

    with col3:
        time_ranges = [
            "00:00–03:00",
            "03:00–06:00",
            "06:00–08:00",
            "08:00–10:00",
            "10:00–12:00",
            "12:00-14:00",
            "14:00–17:00",
            "17:00–19:00",
            "19:00–22:00",
            "22:00–24:00"
        ]

        # 시간대 selectbox
        time_block = st.selectbox(
            "시간",
            options=list(range(1, 11)),
            format_func=lambda x: f"{time_ranges[x - 1]}"
        )

    with col4:
        day_of_week = st.selectbox("요일", [1, 2, 3, 4, 5, 6, 7],
                                   format_func=lambda x: ["월", "화", "수", "목", "금", "토", "일"][x - 1])
    # with col5:
    #     avg_temp = st.slider("평균 기온 (°C)", -5.0, 40.0, 25.0, 0.5)


    predict_button = st.button("🚀 구별 매출 예측 실행", type="primary", use_container_width=True)

    predictions = None
    if predict_button:
        try:
            # 예측 조건 설정
            conditions = {
                'gender_F': 1 if gender == "여성" else 0,
                'gender_M': 1 if gender == "남성" else 0,
                'age_group_code': age_group,
                'time_block_code': time_block,
                'day_of_week': day_of_week
            }

            # 구별 예측 실행
            with st.spinner("예측 중..."):
                predictions = analyzer.predict_by_districts(conditions)
                st.success("✅ 예측 완료!")

        except Exception as e:
            st.error(f"❌ 예측 오류: {str(e)}")

    st.markdown("---")

    # 장안구 / 권선구
    col1, col2 = st.columns([1, 1])

    with col1:
        display_district_info("장안구", data, predictions)
    with col2:
        display_district_info("권선구", data, predictions)

    # 팔달구 / 영통구
    col1, col2 = st.columns([1, 1])

    with col1:
        display_district_info("팔달구", data, predictions)
    with col2:
        display_district_info("영통구", data, predictions)

def display_district_info(district_name: str, data: pd.DataFrame, predictions: dict = None):
    """각 구별 정보 표시 (UI 개선 버전)"""

    st.markdown(f"### {district_name}")

    # 데이터 필터링
    district_data = data[data['district_name'] == district_name]

    if district_data.empty:
        st.warning(f"⚠️ '{district_name}'에 해당하는 데이터가 없습니다.")
        return

    # 상단 매출 요약 카드
    col1, col2, col3 = st.columns(3)
    with col1:
        if predictions and district_name in predictions:
            pred_amount = predictions[district_name]['predicted_amount']
            st.metric("예상 매출", f"{pred_amount:,.0f}원")
    with col2:
        avg_amount = district_data['transaction_amount'].mean()
        st.metric("평균 매출", f"{avg_amount:,.0f}원")
    with col3:
        total_transactions = len(district_data)
        st.metric("총 거래 수", f"{total_transactions:,}건")

    st.markdown("---")

    # 매출 최고 금액 카드
    col_max, col_top_time = st.columns(2)
    with col_max:
        max_amount = district_data['transaction_amount'].max()
        st.metric("최고 매출", f"{max_amount:,.0f}원")

    # 인기 시간대 그래프
    with col_top_time:
        time_sales = district_data.groupby('time_block_code')['transaction_amount'].sum().sort_values(ascending=False)
        top_times = time_sales.head(3).reset_index()
        chart = alt.Chart(top_times).mark_bar().encode(
            x=alt.X('time_block_code:O', title='시간대'),
            y=alt.Y('transaction_amount:Q', title='매출(원)'),
            color=alt.Color('time_block_code:N', legend=None)
        ).properties(height=200, title="⏰ 인기 시간대 TOP 3")
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # 성별 분포 파이 차트
    st.markdown("### 성별 분포")
    gender_dist = district_data['gender'].value_counts().reset_index()
    gender_dist.columns = ['gender', 'count']
    gender_dist['gender_label'] = gender_dist['gender'].map({'F': '여성', 'M': '남성'})
    pie_chart = alt.Chart(gender_dist).mark_arc(innerRadius=40).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="gender_label", type="nominal"),
        tooltip=['gender_label', 'count']
    ).properties(height=250)
    st.altair_chart(pie_chart, use_container_width=True)

    # 연령대 분포 바 차트
    st.markdown("### 연령대별 분포")
    age_dist = district_data['age_group_code'].value_counts().sort_index().reset_index()
    age_dist.columns = ['age_group_code', 'count']
    age_chart = alt.Chart(age_dist).mark_bar().encode(
        x=alt.X("age_group_code:O", title="연령대"),
        y=alt.Y("count:Q", title="고객 수"),
        tooltip=["age_group_code", "count"],
        color=alt.Color("age_group_code:N", legend=None)
    ).properties(height=250)
    st.altair_chart(age_chart, use_container_width=True)

if __name__ == "__main__":
    main()

    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7F8C8D;'>🚀 XGBoost 기반 수원시 구별 매출 예측 시스템</div>",
        unsafe_allow_html=True
    )