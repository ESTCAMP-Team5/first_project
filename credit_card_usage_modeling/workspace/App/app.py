import streamlit as st
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

    st.header("수원시\n카페 업종 매출 예측하기")
    display_temperature_and_time_analysis(data, analyzer)
    st.markdown("---")
    st.header("수원시 행정구별 카페 업종 매출 예측하기")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        # 기본 정보
        gender = st.selectbox("성별", ["여성", "남성"])

        # 예측 버튼

    with col2:
        age_group = st.selectbox("연령대", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], format_func=lambda x: f"연령대 {x}")

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
    with col5:
        avg_temp = st.slider("평균 기온 (°C)", -5.0, 40.0, 25.0, 0.5)

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
                'day_of_week': day_of_week,
                'avg_temp': avg_temp
            }

            # 구별 예측 실행
            with st.spinner("예측 중..."):
                predictions = analyzer.predict_by_districts(conditions)
                st.success("✅ 예측 완료!")

        except Exception as e:
            st.error(f"❌ 예측 오류: {str(e)}")

    st.markdown("---")

    # 2x2 그리드로 각 구 정보 표시
    col1, col2 = st.columns(2)

    with col1:
        display_district_info("장안구", data, predictions)

    with col2:
        display_district_info("권선구", data, predictions)

    # 두 번째 행
    col1, col2 = st.columns(2)

    with col1:
        display_district_info("팔달구", data, predictions)

    with col2:
        display_district_info("영통구", data, predictions)


def display_temperature_and_time_analysis(data, analyzer):
    """기온별 및 시간대별 예측 매출 분석 시각화"""
    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            temp_fig = analyzer.create_temp_analysis_chart(data)
            if temp_fig.data and len(temp_fig.data) > 0:
                st.plotly_chart(temp_fig, use_container_width=True)

                optimal_conditions = analyzer.get_optimal_conditions(data)
                if 'optimal_temp' in optimal_conditions:
                    st.markdown(f"🌡️ **{optimal_conditions['optimal_temp']['message']}**")
            else:
                st.warning("기온별 분석 차트에 데이터가 없습니다.")
                st.write("**디버깅 정보:**")
                st.write(f"- 모델 학습 상태: {analyzer.is_fitted}")
                st.write(f"- 데이터 컬럼: {list(data.columns)}")
                if analyzer.is_fitted:
                    st.write(f"- 모델 피처: {analyzer.feature_names}")
        except Exception as e:
            st.error(f"기온별 분석 오류: {str(e)}")
            st.write("**상세 오류 정보:**")
            st.code(str(e))

    with col2:
        try:
            time_fig = analyzer.create_time_analysis_chart(data)
            if time_fig.data:
                st.plotly_chart(time_fig, use_container_width=True)

                optimal_conditions = analyzer.get_optimal_conditions(data)
                if 'optimal_time' in optimal_conditions:
                    st.markdown(f"⏱️ **{optimal_conditions['optimal_time']['message']}**")
            else:
                st.info("시간대별 분석 데이터가 없습니다.")
        except Exception as e:
            st.error(f"시간대별 분석 오류: {str(e)}")


def display_district_info(district_name, data, predictions):
    district_data = data[data['district_name'] == district_name]

    # 컨테이너로 전체 구역을 감싸서 패딩과 배경 추가
    with st.container():
        st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                    <h2 style=text-align: center; margin: 0; font-weight: bold;">
                        📍 {district_name} 분석
                    </h2>
                </div>
                """, unsafe_allow_html=True)

        # 메인 그리드 레이아웃 (벤토 스타일)
        # 첫 번째 행: 큰 예상매출 카드 + 두 개의 작은 카드
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # 큰 예상매출 카드
            if predictions and district_name in predictions:
                pred_amount = predictions[district_name]['predicted_amount']
                st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); 
                                   padding: 30px; border-radius: 20px; text-align: leading; 
                                   box-shadow: 0 8px 32px rgba(255,107,107,0.3); height: 160px;
                                   display: flex; flex-direction: column; justify-content: leading;">
                            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.2em;">
                                예상 매출
                            </p>
                            <h1 style=" margin: 0; font-size: 2.5em; font-weight: bold;">
                                {pred_amount:,.0f}원
                            </h1>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #95a5a6, #bdc3c7); 
                                   padding: 30px; border-radius: 20px; text-align: leading; 
                                   height: 160px; display: flex; flex-direction: column; justify-content: leading;">
                            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
                                예상 매출
                            </p>
                            <h1 style="margin: 0; font-size: 1.8em;">
                                검색 필요
                            </h1>
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            # 평균 매출 카드
            avg_amount = district_data['transaction_amount'].mean()
            st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); 
                               padding: 20px; border-radius: 15px; text-align: leading; 
                               box-shadow: 0 4px 20px rgba(78,205,196,0.3); height: 160px;
                               display: flex; flex-direction: column; justify-content: leading;">
                        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
                            평균 매출
                        </p>
                        <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                            {avg_amount:,.0f}원
                        </h2>

                    </div>
                    """, unsafe_allow_html=True)

        with col3:
            # 총 거래수 카드
            total_transactions = len(district_data)
            st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #A8E6CF, #7FCDCD); 
                               padding: 20px; border-radius: 15px; text-align: leading; 
                               box-shadow: 0 4px 20px rgba(168,230,207,0.3); height: 160px;
                               display: flex; flex-direction: column; justify-content: leading;">
                        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
                            총 거래수
                        </p>
                         <h2 style=" margin: 0; font-size: 1.8em; font-weight: bold;">
                            {total_transactions:,}건
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 두 번째 행: 최고매출 + 인기시간대 차트
        col1, col2 = st.columns([1, 2])

        with col1:
            # 최고 매출 카드
            max_amount = district_data['transaction_amount'].max()
            st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #FFD93D, #FF8C42); 
                               padding: 25px; border-radius: 18px; text-align: leading; 
                               box-shadow: 0 6px 25px rgba(255,217,61,0.3); height: 200px;
                               display: flex; flex-direction: column; justify-content: leading;">
                        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1em;">
                            최고 매출 기록
                        </p>
                        <h2 style=" margin: 10px 0 5px 0; font-size: 1.6em;">
                            {max_amount:,.0f}원
                        </h2>

                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            # 인기 시간대 차트 (배경 추가)
            st.markdown("""
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); 
                               padding: 20px; border-radius: 18px; 
                               box-shadow: 0 6px 25px rgba(102,126,234,0.3);">
                    """, unsafe_allow_html=True)

            st.markdown("<h3 style='text-align: center; margin-top: 0;'>인기 시간대 TOP 3</h3>",
                        unsafe_allow_html=True)

            time_sales = district_data.groupby('time_block_code')['transaction_amount'].sum().sort_values(
                ascending=False)
            top_times = time_sales.head(3).reset_index()

            chart = alt.Chart(top_times).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X('time_block_code:O', title='시간대', axis=alt.Axis()),
                y=alt.Y('transaction_amount:Q', title='매출(원)',
                        axis=alt.Axis()),
                color=alt.Color('time_block_code:N',
                                scale=alt.Scale(range=['#FF6B6B', '#4ECDC4', '#FFD93D']),
                                legend=None),
                tooltip=['time_block_code', 'transaction_amount']
            ).properties(
                height=120,
                background='transparent'
            ).configure_axis(
                grid=False,
                domain=False
            ).configure_view(
                strokeWidth=0
            )

            st.altair_chart(chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 세 번째 행: 성별 분포 + 연령대 분포
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                    <div style="background: linear-gradient(45deg, #FA8072, #F0E68C); 
                               padding: 20px; border-radius: 18px; 
                               box-shadow: 0 6px 25px rgba(250,128,114,0.3);">
                    """, unsafe_allow_html=True)

            st.markdown("<h3 style='text-align: center; margin-top: 0;'>성별 분포</h3>",
                        unsafe_allow_html=True)

            gender_dist = district_data['gender'].value_counts().reset_index()
            gender_dist.columns = ['gender', 'count']
            gender_dist['gender_label'] = gender_dist['gender'].map({'F': '여성', 'M': '남성'})

            pie_chart = alt.Chart(gender_dist).mark_arc(
                innerRadius=50,
                strokeWidth=3
            ).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(
                    field="gender",
                    type="nominal",
                    scale=alt.Scale(range=['#FF69B4', '#4169E1']),
                    legend=alt.Legend(orient='bottom')
                ),
                tooltip=['gender', 'count']
            ).properties(
                height=200,
                background='transparent'
            ).configure_legend(
            )

            st.altair_chart(pie_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                    <div style="background: linear-gradient(45deg, #9370DB, #20B2AA); 
                               padding: 20px; border-radius: 18px; 
                               box-shadow: 0 6px 25px rgba(147,112,219,0.3);">
                    """, unsafe_allow_html=True)

            st.markdown("<h3 style='text-align: center; margin-top: 0;'>연령대별 분포</h3>",
                        unsafe_allow_html=True)

            age_dist = district_data['age_group_code'].value_counts().sort_index().reset_index()
            age_dist.columns = ['age_group_code', 'count']

            age_chart = alt.Chart(age_dist).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8
            ).encode(
                x=alt.X("age_group_code:O",
                        title="연령대",
                        axis=alt.Axis()),
                y=alt.Y("count:Q",
                        title="고객 수",
                        axis=alt.Axis()),
                color=alt.Color(
                    "age_group_code:N",
                    scale=alt.Scale(range=['#FF6B6B', '#4ECDC4', '#FFD93D', '#FF8C42', '#A8E6CF']),
                    legend=None
                ),
                tooltip=["age_group_code", "count"]
            ).properties(
                height=200,
                background='transparent'
            ).configure_axis(
                grid=False,
                domain=False
            ).configure_view(
                strokeWidth=0
            )

            st.altair_chart(age_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()