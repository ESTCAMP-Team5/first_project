# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# 로컬 모듈 import
from data_repository import create_data_repository
from models import CoffeeSalesAnalyzer, BusinessInsightGenerator

# 페이지 설정
st.set_page_config(
    page_title="Suwon City Coffee Sales Forecast & Insights",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #2C3E50, #34495E);
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }

    .insight-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }

    .metric-card {
        background-color: #2C3E50;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #34495E;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(repo_type: str = "mock"):
    """데이터 로드 및 캐싱"""
    repo = create_data_repository(repo_type)
    return repo.get_combined_data()


@st.cache_resource
def load_analyzer():
    """분석기 로드 및 캐싱"""
    return CoffeeSalesAnalyzer()


def main():
    # 헤더
    st.markdown('<h1 class="main-header">☕ Suwon City Coffee Sales Forecast & Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze and predict coffee sales based on temperature and time patterns</p>',
                unsafe_allow_html=True)

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ Settings")

        # 데이터 소스 선택
        data_source = st.selectbox(
            "Data Source",
            ["mock", "sqlite"],
            help="Choose your data source"
        )

        # 날짜 범위 선택
        st.subheader("📅 Date Range")
        date_range = st.date_input(
            "Select date range",
            value=(datetime(2024, 1, 1), datetime(2024, 3, 31)),
            help="Choose the analysis period"
        )

        # 분석 옵션
        st.subheader("📊 Analysis Options")
        show_insights = st.checkbox("Show Business Insights", value=True)
        show_forecast = st.checkbox("Show Forecast", value=True)
        forecast_days = st.slider("Forecast Days", 1, 14, 7)

    try:
        # 데이터 로드
        with st.spinner("Loading data..."):
            data = load_data(data_source)

            # 날짜 필터링
            if len(date_range) == 2:
                start_date, end_date = date_range
                data = data[
                    (data['date'].dt.date >= start_date) &
                    (data['date'].dt.date <= end_date)
                    ]

        # 분석기 로드 및 모델 학습
        with st.spinner("Training models..."):
            analyzer = load_analyzer()
            analyzer.fit_models(data)

        # 메인 대시보드
        display_main_dashboard(data, analyzer)

        # 비즈니스 인사이트
        if show_insights:
            display_business_insights(data, analyzer)

        # 예측 섹션
        if show_forecast:
            display_forecast_section(data, analyzer, forecast_days)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check your data source configuration.")


def display_main_dashboard(data: pd.DataFrame, analyzer: CoffeeSalesAnalyzer):
    """메인 대시보드 표시"""

    # 첫 번째 행: 데이터 미리보기, 산점도, 메트릭
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("📊 Data Preview")
        preview_data = data.head(10).copy()
        preview_data['date'] = preview_data['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(
            preview_data[['date', 'avg_temperature', 'transaction_count']],
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("🌡️ Temperature vs. Coffee Sales")
        fig_scatter = analyzer.create_temperature_sales_plot(data)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col3:
        st.subheader("📈 Daily Trends")

        # 상관관계 분석
        temp_corr = analyzer.get_temperature_correlation_analysis(data)

        col3_1, col3_2 = st.columns(2)

        with col3_1:
            st.metric(
                label="R²",
                value=f"{temp_corr['r_squared']:.3f}",
                help="결정계수 - 모델의 설명력"
            )

        with col3_2:
            st.metric(
                label="P-value",
                value=f"{temp_corr['p_value']:.3f}",
                help="통계적 유의성"
            )

        # 모델 성능 메트릭
        if analyzer.is_fitted:
            st.subheader("🤖 Model Performance")
            metrics = analyzer.get_model_metrics()

            for model_name, model_metrics in metrics.items():
                st.write(f"**{model_name.title()} Model:**")
                st.write(f"- R² Score: {model_metrics['r2_score']:.3f}")
                st.write(f"- MAE: {model_metrics['mae']:.0f}")
                st.write(f"- RMSE: {model_metrics['rmse']:.0f}")

    # 두 번째 행: 시계열, 히트맵
    st.markdown("---")
    col4, col5 = st.columns([1, 1])

    with col4:
        st.subheader("📊 Temperature vs. Coffee Sales (Time Series)")
        fig_time = analyzer.create_time_series_plot(data)
        st.plotly_chart(fig_time, use_container_width=True)

    with col5:
        st.subheader("🗓️ Monthly & Weekly Heatmap")
        fig_heatmap = analyzer.create_heatmap(data)
        st.plotly_chart(fig_heatmap, use_container_width=True)


def display_business_insights(data: pd.DataFrame, analyzer: CoffeeSalesAnalyzer):
    """비즈니스 인사이트 표시"""

    st.markdown("---")
    st.header("💡 Business Insights")

    # 인사이트 생성
    insights = BusinessInsightGenerator.generate_insights(data, analyzer)

    # 요약 통계
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Avg Daily Sales",
            f"{insights['summary']['avg_daily_sales']:,}",
            help="Average daily transactions"
        )

    with col2:
        st.metric(
            "Peak Sales",
            f"{insights['summary']['peak_sales']['sales']:,}",
            help=f"Peak sales on {insights['summary']['peak_sales']['date']}"
        )

    with col3:
        st.metric(
            "Temperature Correlation",
            f"{insights['temperature_insights']['correlation_coefficient']:.3f}",
            help=f"{insights['temperature_insights']['correlation_strength']} correlation"
        )

    with col4:
        weekend_premium = insights['time_insights']['weekend_vs_weekday']['weekend_premium']
        st.metric(
            "Weekend Premium",
            f"{weekend_premium:+.1f}%",
            help="Weekend sales vs weekday sales"
        )

    # 상세 인사이트
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("🌡️ Temperature Insights")
        st.markdown(f"""
        <div class="insight-card">
        <strong>Correlation Strength:</strong> {insights['temperature_insights']['correlation_strength']}<br>
        <strong>Optimal Temperature Range:</strong> {insights['temperature_insights']['optimal_temperature_range']['range']}<br>
        <strong>Avg Sales in Optimal Range:</strong> {insights['temperature_insights']['optimal_temperature_range']['avg_sales']:,}
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.subheader("📅 Time Pattern Insights")
        time_insights = insights['time_insights']
        st.markdown(f"""
        <div class="insight-card">
        <strong>Best Weekday:</strong> {time_insights['best_weekday']}<br>
        <strong>Worst Weekday:</strong> {time_insights['worst_weekday']}<br>
        <strong>Weekend Avg:</strong> {time_insights['weekend_vs_weekday']['weekend_avg']:,}<br>
        <strong>Weekday Avg:</strong> {time_insights['weekend_vs_weekday']['weekday_avg']:,}
        </div>
        """, unsafe_allow_html=True)

    # 권장사항
    st.subheader("🎯 Marketing Recommendations")
    for i, recommendation in enumerate(insights['recommendations'], 1):
        st.markdown(f"{i}. {recommendation}")


def display_forecast_section(data: pd.DataFrame, analyzer: CoffeeSalesAnalyzer, forecast_days: int):
    """예측 섹션 표시"""

    st.markdown("---")
    st.header("🔮 Sales Forecast")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📈 Forecast Results")

        if analyzer.is_fitted:
            # 예측 생성
            forecast_df = analyzer.generate_forecast(data, forecast_days)

            # 예측 데이터 표시
            display_forecast = forecast_df.copy()
            display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m-%d')
            display_forecast['day_name'] = forecast_df['date'].dt.strftime('%A')

            st.dataframe(
                display_forecast[['date', 'day_name', 'predicted_temperature', 'predicted_sales']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'date': 'Date',
                    'day_name': 'Day',
                    'predicted_temperature': st.column_config.NumberColumn(
                        'Temp (°C)', format="%.1f"
                    ),
                    'predicted_sales': st.column_config.NumberColumn(
                        'Predicted Sales', format="%d"
                    )
                }
            )

            # 예측 요약
            total_predicted = forecast_df['predicted_sales'].sum()
            avg_predicted = forecast_df['predicted_sales'].mean()

            st.metric("Total Predicted Sales", f"{total_predicted:,}")
            st.metric("Avg Daily Predicted Sales", f"{avg_predicted:.0f}")

        else:
            st.warning("Model not fitted. Please check your data.")

    with col2:
        st.subheader("🎯 Manual Prediction")

        # 수동 예측 입력
        st.write("Enter conditions for custom prediction:")

        pred_temp = st.slider("Temperature (°C)", -5.0, 35.0, 15.0, 0.1)
        pred_day = st.selectbox("Day of Week",
                                ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                 'Friday', 'Saturday', 'Sunday'])
        pred_month = st.selectbox("Month",
                                  ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'])

        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        month_mapping = {month: i + 1 for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June',
                                                                 'July', 'August', 'September', 'October', 'November',
                                                                 'December'])}

        if st.button("Predict Sales", type="primary"):
            if analyzer.is_fitted:
                day_num = day_mapping[pred_day]
                month_num = month_mapping[pred_month]
                is_weekend = day_num >= 5

                prediction = analyzer.predict_sales(pred_temp, day_num, month_num, is_weekend)

                st.success(f"Predicted Sales: **{prediction['ensemble_prediction']:,}** transactions")

                # 모델별 예측 비교
                st.write("**Model Comparison:**")
                st.write(f"- Linear Model: {prediction['linear_prediction']:,}")
                st.write(f"- Random Forest: {prediction['random_forest_prediction']:,}")
                st.write(f"- Ensemble: {prediction['ensemble_prediction']:,}")
            else:
                st.error("Model not fitted!")


def display_detailed_data_view(data: pd.DataFrame):
    """상세 데이터 뷰 표시"""

    st.markdown("---")
    st.subheader("📋 Detailed Data View")

    # 필터링 옵션
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        temp_range = st.slider(
            "Temperature Range (°C)",
            min_value=float(data['avg_temperature'].min()),
            max_value=float(data['avg_temperature'].max()),
            value=(float(data['avg_temperature'].min()), float(data['avg_temperature'].max())),
            step=0.1
        )

    with col_filter2:
        transaction_range = st.slider(
            "Transaction Range",
            min_value=int(data['transaction_count'].min()),
            max_value=int(data['transaction_count'].max()),
            value=(int(data['transaction_count'].min()), int(data['transaction_count'].max())),
            step=100
        )

    # 필터링된 데이터
    filtered_data = data[
        (data['avg_temperature'] >= temp_range[0]) &
        (data['avg_temperature'] <= temp_range[1]) &
        (data['transaction_count'] >= transaction_range[0]) &
        (data['transaction_count'] <= transaction_range[1])
        ].copy()

    filtered_data['date'] = filtered_data['date'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()

    # 상세 데이터 뷰 (항상 표시)
    try:
        data = load_data("mock")
        display_detailed_data_view(data)
    except:
        pass

    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #7F8C8D; font-size: 0.9rem;'>
        🚀 Built with Streamlit | 📊 Modular Architecture with Repository Pattern
        </div>
        """,
        unsafe_allow_html=True
    )