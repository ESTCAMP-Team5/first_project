# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
# 로컬 모듈 import
from data_repository import create_data_repository
from models import CoffeeSalesAnalyzer, BusinessInsightGenerator

# 페이지 설정
st.set_page_config(
    page_title="수원시 커피 업종 매출에 영향을 미치는 요인",
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
    st.markdown('<h1 class="main-header">☕ 수원시 커피 업종 매출에 영향을 미치는 요인</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">여러 외부 요인들에 의해 커피 매출에 어떠한 영향을 미치는지 분석 및 예측</p>',
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
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("평균 기온에 따른 예측 매출")

        # 데이터 로드 및 전처리
        df = pd.read_csv('C:/Users/user-pc/Desktop/수원시 24.3~25.3/df_coffee_권선구.csv')
        df = df.drop(columns=['Unnamed: 0'])
        df = df.dropna()
        df_encoded = pd.get_dummies(df, columns=['sex']).astype(float)
        df_encoded['ta_ymd'] = pd.to_datetime(df_encoded['ta_ymd'])

        # 고객 기준 누적 지표 계산
        df_encoded = df_encoded.sort_values(by=['sex_F','sex_M', 'age', 'ta_ymd'])
        df_encoded['past_7day_amt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        df_encoded['past_30day_cnt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['cnt'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
        df_encoded['mean_amt_per_visit'] = df_encoded['amt'] / (df_encoded['cnt'] + 1e-5)

        # 피처/타겟 설정
        features_amt = ['sex_F','sex_M', 'age', 'hour', 'day',
                        'past_7day_amt', 'past_30day_cnt', 'mean_amt_per_visit', 'AvgTemp', 'cty_rgn_no']
        X = df_encoded[features_amt]
        y_amt = df_encoded['amt']
        X_amt_train, X_amt_test, y_amt_train, y_amt_test = train_test_split(X, y_amt, test_size=0.2, random_state=42)

        # 최적 하이퍼파라미터 기반 모델
        best_params_LGBM = {
            'num_leaves': 89, 'max_depth': 15, 'learning_rate': 0.0837, 'n_estimators': 859,
            'min_child_samples': 84, 'subsample': 0.9619, 'colsample_bytree': 0.7480,
            'reg_alpha': 3.8135, 'reg_lambda': 1.5787
        }

        best_params_XGBoost = {
            'n_estimators': 645, 'max_depth': 14, 'learning_rate': 0.03498956134885634,
            'subsample': 0.7132225654522615, 'colsample_bytree': 0.8050397258381967,
            'gamma': 0.3573772428817774, 'reg_alpha': 3.562730391807268, 'reg_lambda': 0.511245824338082
        }

        model_lgbm = LGBMRegressor(**best_params_LGBM)
        model_xgb = XGBRegressor(**best_params_XGBoost)

        # 앙상블 모델 학습 및 예측
        temp_model = VotingRegressor(estimators=[
            ('LGBM', model_lgbm),
            ('XGB', model_xgb)
        ], weights=[0.2, 0.8])
        temp_model.fit(X_amt_train, y_amt_train)
        y_pred_temp_amt = temp_model.predict(X_amt_test)

        # 기온별 평균 예측 매출 계산 및 시각화
        X_amt_test_copy = X_amt_test.copy()
        X_amt_test_copy['predicted_amt'] = y_pred_temp_amt
        mean_amt_by_temp = X_amt_test_copy.groupby('AvgTemp')['predicted_amt'].mean()

        fig = go.Figure()

        fig.add_trace(go.Bar(
        x=mean_amt_by_temp.index,
        y=mean_amt_by_temp.values,
        text=[f"{int(y):,}원" for y in mean_amt_by_temp.values],
        textposition='outside',
        marker_color='lightcoral',
        name='예측 매출'
        ))

        fig.update_layout(
        title='평균 기온별 평균 예측 매출',
        xaxis=dict(title='평균 기온 (℃)', range=[-1, 30]),
        yaxis=dict(title='예측 매출'),
        showlegend=False,
        bargap=0.15,
        height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 최적 기온 표시
        max_temp = mean_amt_by_temp.idxmax()
        max_amt = mean_amt_by_temp.max()
        st.markdown(f"🌡️ **기온이 {max_temp:.1f}℃일 때 예측 매출이 가장 높습니다: 약 {max_amt:,.0f}원**")

    with col2:
        st.subheader("시간대에 따른 예측 매출")

        # 데이터 로드 및 전처리
        df = pd.read_csv('C:/Users/user-pc/Desktop/수원시 24.3~25.3/df_coffee_권선구.csv')
        df = df.drop(columns=['Unnamed: 0'])
        df = df.dropna()
        df_encoded = pd.get_dummies(df, columns=['sex']).astype(float)
        df_encoded['ta_ymd'] = pd.to_datetime(df_encoded['ta_ymd'])

        # 고객 기준 누적 지표 계산
        df_encoded = df_encoded.sort_values(by=['sex_F','sex_M', 'age', 'ta_ymd'])
        df_encoded['past_7day_amt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        df_encoded['past_30day_cnt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['cnt'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
        df_encoded['mean_amt_per_visit'] = df_encoded['amt'] / (df_encoded['cnt'] + 1e-5)

        # 피처/타겟 설정
        features_amt = ['sex_F','sex_M', 'age', 'hour', 'day',
                        'past_7day_amt', 'past_30day_cnt', 'mean_amt_per_visit', 'AvgTemp', 'cty_rgn_no']
        X = df_encoded[features_amt]
        y_amt = df_encoded['amt']
        X_amt_train, X_amt_test, y_amt_train, y_amt_test = train_test_split(X, y_amt, test_size=0.2, random_state=42)

        # 최적 하이퍼파라미터 기반 모델
        best_params_LGBM = {
            'num_leaves': 89, 'max_depth': 15, 'learning_rate': 0.0837, 'n_estimators': 859,
            'min_child_samples': 84, 'subsample': 0.9619, 'colsample_bytree': 0.7480,
            'reg_alpha': 3.8135, 'reg_lambda': 1.5787
        }

        best_params_XGBoost = {
            'n_estimators': 645, 'max_depth': 14, 'learning_rate': 0.03498956134885634,
            'subsample': 0.7132225654522615, 'colsample_bytree': 0.8050397258381967,
            'gamma': 0.3573772428817774, 'reg_alpha': 3.562730391807268, 'reg_lambda': 0.511245824338082
        }

        model_lgbm = LGBMRegressor(**best_params_LGBM)
        model_xgb = XGBRegressor(**best_params_XGBoost)

        # 앙상블 모델 학습 및 예측
        hour_model = VotingRegressor(estimators=[
            ('LGBM', model_lgbm),
            ('XGB', model_xgb)
        ], weights=[0.2, 0.8])
        hour_model.fit(X_amt_train, y_amt_train)
        y_pred_hour_amt = hour_model.predict(X_amt_test)

        # 기온별 평균 예측 매출 계산 및 시각화
        X_amt_test_copy = X_amt_test.copy()
        X_amt_test_copy['predicted_amt'] = y_pred_hour_amt
        mean_amt_by_hour = X_amt_test_copy.groupby('hour')['predicted_amt'].mean()

        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.barplot(x=mean_amt_by_hour.index.astype(int), y=mean_amt_by_hour.values, ax=ax)
        # ax.set_title("시간대별 평균 예측 매출")
        # ax.set_xlabel("시간대 (단위 예: 1=2시간24분씩)")
        # ax.set_ylabel("예측 매출")
        # for x, y in zip(mean_amt_by_hour.index.astype(int), mean_amt_by_hour.values):
        #     rounded_y = round(int(y),-2)
        #     ax.text(x-1, y+1500, f'{rounded_y:,}\\', ha='center', fontsize=9)
        
        # st.pyplot(fig)
        # st.text("오후 12:00 ~ 14:24에 매출이 {:,}원으로 가장 높았다.".format(round(int(mean_amt_by_hour.values.max()),-2)))
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=mean_amt_by_hour.index.astype(int),
            y=mean_amt_by_hour.values,
            text=[f"{round(int(y), -2):,}원" for y in mean_amt_by_hour.values],
            textposition='outside',
            marker_color='skyblue',
            name='예측 매출'
            ))

        fig.update_layout(
        title='시간대별 평균 예측 매출',
        xaxis=dict(
        title='시간대 (단위 예: 1=2시간24분씩)',
        tickmode='linear',
        dtick=1
        ),
        yaxis=dict(title='예측 매출'),
        showlegend=False,
        bargap=0.2,
        height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 최고 매출 시간 출력
        peak_hour = mean_amt_by_hour.idxmax()
        peak_value = round(int(mean_amt_by_hour.max()), -2)
        st.markdown(f"⏱ 12:00 ~ 14:24에 매출이 {peak_value:,}원으로 가장 높았다.")

    # 두 번째 행: 시계열, 히트맵
    st.markdown("---")
    col4, col5 = st.columns([1, 1])
    
    with col4:
        # features_amt = ['amt', 'cnt', 'age', 'AvgTemp', 'hour', 'day', 
        #         'cty_rgn_no']

        # # 데이터 불러오기 및 전처리
        # df = pd.read_csv('C:/Users/user-pc/Desktop/수원시 24.3~25.3/df_coffee_권선구.csv')
        # df = df.drop(columns=['Unnamed: 0'])
        # df = df.dropna()
        # df_encoded = pd.get_dummies(df, columns=['sex']).astype(float)
        # df_encoded = df_encoded.sort_values(by=['sex_F','sex_M', 'age', 'ta_ymd'])
        # df_encoded['past_7day_amt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        # df_encoded['past_30day_cnt'] = df_encoded.groupby(['sex_F','sex_M', 'age'])['cnt'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
        # df_encoded['mean_amt_per_visit'] = df_encoded['amt'] / (df_encoded['cnt'] + 1e-5)

        # X = df_encoded[features_amt]
        # y = LabelEncoder().fit_transform(df_encoded['sex_F'])

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # sex_model = LGBMClassifier()
        # sex_model.fit(X_train, y_train)

        # # Streamlit UI
        # st.subheader("🧍 고객 정보 기반 성별 예측")

        # st.markdown("💡 아래 수치를 입력하면 고객의 성별을 예측합니다.")

        # st.markdown("amt : 매출금액, cnt : 구매횟수, age : 연령대(1 ~ 9), AvgTemp : 평균 기온 \n" \
        # "\nhour : 시간대(1 ~ 10), day : 요일(1 ~ 7), cty_rgn_no : 행정구코드 \n" \
        # "\n41111 : 장안구 41113 : 권선구 41115 : 팔달구 41117 : 영통구")

        # # 사용자 입력 받기
        # if 'inputs' not in st.session_state:
        #     st.session_state.inputs = {}
        # for feature in features_amt:
        #     if feature == "cty_rgn_no":
        #         st.session_state.inputs[feature] = st.selectbox("cty_rgn_no (행정구역 코드 선택)", [41111, 41113, 41115, 41117])
        #     elif feature == "AvgTemp":
        #         st.session_state.inputs[feature] = st.number_input("AvgTemp (평균 기온)", value=15.0, step=0.1, format="%.1f")
        #     else:
        #         value = st.text_input(f"{feature} (정수 입력)", value=st.session_state.inputs.get(feature, ""))
        #     try:
        #         st.session_state.inputs[feature] = int(value)
        #     except ValueError:
        #         st.warning(f"❗ '{feature}' 값은 정수로 입력되어야 합니다.")
        #         st.stop()  # 유효하지 않으면 예측까지 가지 않음

        # # 예측 버튼
        # if st.button("성별 예측하기"):
        #     input_list = [st.session_state.inputs[feature] for feature in features_amt]
        #     input_array = np.array(input_list).reshape(1, -1)
        #     pred = sex_model.predict(input_array)[0]
        #     prob = sex_model.predict_proba(input_array)[0][1]

        #     result = "여자 👩" if pred == 1 else "남자 👨"
        #     st.success(f"예측 성별: **{result}**")
        #     st.metric("여자일 확률", f"{prob:.2%}")
        #     st.metric("남자일 확률", f"{(1 - prob):.2%}")
        pass

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