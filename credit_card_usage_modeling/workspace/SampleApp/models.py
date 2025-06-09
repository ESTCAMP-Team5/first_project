import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Dict, Tuple, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


class CoffeeSalesAnalyzer:
    """커피 판매 분석을 위한 메인 클래스"""

    def __init__(self):
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_fitted = False
        self.model_metrics = {}

    def fit_models(self, data: pd.DataFrame):
        """모델 학습"""
        # 특성 생성
        X, y = self._prepare_features(data)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 모델 학습
        self.linear_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)

        # 성능 평가
        self._evaluate_models(X_test, y_test)
        self.is_fitted = True

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """특성 준비"""
        features = []

        # 온도 특성
        features.append(data['avg_temperature'].values)

        # 시간 특성
        features.append(data['day_of_week'].values)
        features.append(data['month'].values)
        features.append(data['is_weekend'].astype(int).values)

        # 계절성 특성
        day_of_year = data['date'].dt.dayofyear
        features.append(np.sin(2 * np.pi * day_of_year / 365))  # 계절성
        features.append(np.cos(2 * np.pi * day_of_year / 365))

        X = np.column_stack(features)
        y = data['transaction_count'].values

        return X, y

    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """모델 성능 평가"""
        models = {
            'linear': self.linear_model,
            'random_forest': self.rf_model
        }

        for name, model in models.items():
            y_pred = model.predict(X_test)

            self.model_metrics[name] = {
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }

    def get_temperature_correlation_analysis(self, data: pd.DataFrame) -> Dict:
        """온도와 판매량 상관관계 분석"""
        temp = data['avg_temperature']
        sales = data['transaction_count']

        # 통계 분석
        slope, intercept, r_value, p_value, std_err = stats.linregress(temp, sales)

        return {
            'correlation_coefficient': np.corrcoef(temp, sales)[0, 1],
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'std_error': std_err
        }

    def predict_sales(self, temperature: float, day_of_week: int,
                      month: int, is_weekend: bool = False) -> Dict:
        """판매량 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit_models()를 먼저 호출하세요.")

        # 특성 준비
        day_of_year = (month - 1) * 30 + 15  # 대략적인 계산
        features = np.array([[
            temperature,
            day_of_week,
            month,
            int(is_weekend),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365)
        ]])

        # 예측
        linear_pred = self.linear_model.predict(features)[0]
        rf_pred = self.rf_model.predict(features)[0]

        return {
            'linear_prediction': int(linear_pred),
            'random_forest_prediction': int(rf_pred),
            'ensemble_prediction': int((linear_pred + rf_pred) / 2)
        }

    def generate_forecast(self, data: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """미래 판매량 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        last_date = data['date'].max()
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]

        # 온도 예측 (단순화: 최근 평균 + 노이즈)
        recent_temp_mean = data['avg_temperature'].tail(7).mean()
        recent_temp_std = data['avg_temperature'].tail(7).std()
        predicted_temps = np.random.normal(recent_temp_mean, recent_temp_std, days_ahead)

        forecasts = []
        for i, (date, temp) in enumerate(zip(forecast_dates, predicted_temps)):
            day_of_week = date.weekday()
            month = date.month
            is_weekend = day_of_week >= 5

            prediction = self.predict_sales(temp, day_of_week, month, is_weekend)

            forecasts.append({
                'date': date,
                'predicted_temperature': round(temp, 1),
                'predicted_sales': prediction['ensemble_prediction'],
                'day_of_week': day_of_week,
                'is_weekend': is_weekend
            })

        return pd.DataFrame(forecasts)

    def get_model_metrics(self) -> Dict:
        """모델 성능 메트릭 반환"""
        return self.model_metrics

    def create_temperature_sales_plot(self, data: pd.DataFrame) -> go.Figure:
        """온도-판매량 산점도 생성"""
        fig = px.scatter(
            data,
            x='avg_temperature',
            y='transaction_count',
            title="Temperature vs Coffee Sales Relationship",
            labels={'avg_temperature': 'Average Temperature (°C)',
                    'transaction_count': 'Daily Transactions'},
            template='plotly_dark'
        )

        # 회귀선 추가
        z = np.polyfit(data['avg_temperature'], data['transaction_count'], 1)
        p = np.poly1d(z)
        fig.add_traces(go.Scatter(
            x=data['avg_temperature'],
            y=p(data['avg_temperature']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2)
        ))

        return fig

    def create_time_series_plot(self, data: pd.DataFrame) -> go.Figure:
        """시계열 그래프 생성"""
        fig = go.Figure()

        # 온도 라인
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['avg_temperature'],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))

        # 거래량 라인
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['transaction_count'],
            mode='lines',
            name='Transactions',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))

        # 레이아웃 설정
        fig.update_layout(
            title='Temperature and Transaction Trends Over Time',
            xaxis_title='Date',
            yaxis=dict(
                title='Temperature (°C)',
                side='left',
                color='blue'
            ),
            yaxis2=dict(
                title='Daily Transactions',
                side='right',
                overlaying='y',
                color='orange'
            ),
            template='plotly_dark',
            height=400
        )

        return fig

    def create_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """히트맵 생성"""
        # 월별, 요일별 평균 거래량
        heatmap_data = data.groupby(['month', 'day_of_week'])['transaction_count'].mean().unstack()

        # 월 이름 매핑
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # 존재하는 월만 표시
        existing_months = sorted(heatmap_data.index.tolist())
        month_labels = [month_names[m - 1] for m in existing_months]

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Weekday", y="Month", color="Avg Transactions"),
            x=weekday_names,
            y=month_labels,
            color_continuous_scale='Blues',
            title="Average sales by month and weekday"
        )

        fig.update_layout(
            template='plotly_dark',
            height=400
        )

        return fig


class BusinessInsightGenerator:
    """비즈니스 인사이트 생성 클래스"""

    @staticmethod
    def generate_insights(data: pd.DataFrame, analyzer: CoffeeSalesAnalyzer) -> Dict:
        """종합적인 비즈니스 인사이트 생성"""

        # 기본 통계
        avg_daily_sales = data['transaction_count'].mean()
        peak_sales_day = data.loc[data['transaction_count'].idxmax()]
        low_sales_day = data.loc[data['transaction_count'].idxmin()]

        # 온도 분석
        temp_corr = analyzer.get_temperature_correlation_analysis(data)

        # 요일별 분석
        weekday_avg = data.groupby('day_of_week')['transaction_count'].mean()
        best_weekday = weekday_avg.idxmax()
        worst_weekday = weekday_avg.idxmin()

        # 주말 vs 평일
        weekend_avg = data[data['is_weekend']]['transaction_count'].mean()
        weekday_sales_avg = data[~data['is_weekend']]['transaction_count'].mean()

        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                         'Friday', 'Saturday', 'Sunday']

        insights = {
            'summary': {
                'avg_daily_sales': int(avg_daily_sales),
                'total_days_analyzed': len(data),
                'peak_sales': {
                    'date': peak_sales_day['date'].strftime('%Y-%m-%d'),
                    'sales': int(peak_sales_day['transaction_count']),
                    'temperature': round(peak_sales_day['avg_temperature'], 1)
                },
                'low_sales': {
                    'date': low_sales_day['date'].strftime('%Y-%m-%d'),
                    'sales': int(low_sales_day['transaction_count']),
                    'temperature': round(low_sales_day['avg_temperature'], 1)
                }
            },
            'temperature_insights': {
                'correlation_strength': 'Strong' if abs(temp_corr['correlation_coefficient']) > 0.7
                else 'Moderate' if abs(temp_corr['correlation_coefficient']) > 0.5
                else 'Weak',
                'correlation_coefficient': round(temp_corr['correlation_coefficient'], 3),
                'optimal_temperature_range': BusinessInsightGenerator._find_optimal_temp_range(data)
            },
            'time_insights': {
                'best_weekday': weekday_names[best_weekday],
                'worst_weekday': weekday_names[worst_weekday],
                'weekend_vs_weekday': {
                    'weekend_avg': int(weekend_avg),
                    'weekday_avg': int(weekday_sales_avg),
                    'weekend_premium': round((weekend_avg / weekday_sales_avg - 1) * 100, 1)
                }
            },
            'recommendations': BusinessInsightGenerator._generate_recommendations(data, temp_corr)
        }

        return insights

    @staticmethod
    def _find_optimal_temp_range(data: pd.DataFrame) -> Dict:
        """최적 온도 범위 찾기"""
        # 온도를 구간별로 나누어 평균 판매량 계산
        data_copy = data.copy()
        data_copy['temp_range'] = pd.cut(data_copy['avg_temperature'], bins=5)
        temp_avg_sales = data_copy.groupby('temp_range')['transaction_count'].mean()

        optimal_range = temp_avg_sales.idxmax()

        return {
            'range': f"{optimal_range.left:.1f}°C - {optimal_range.right:.1f}°C",
            'avg_sales': int(temp_avg_sales.max())
        }

    @staticmethod
    def _generate_recommendations(data: pd.DataFrame, temp_corr: Dict) -> List[str]:
        """마케팅 권장사항 생성"""
        recommendations = []

        # 온도 기반 권장사항
        if temp_corr['correlation_coefficient'] > 0.5:
            recommendations.append("🌡️ 온도가 높을수록 판매량이 증가합니다. 더운 날씨 예보 시 재고를 충분히 준비하세요.")
        elif temp_corr['correlation_coefficient'] < -0.5:
            recommendations.append("❄️ 온도가 낮을수록 판매량이 증가합니다. 추운 날씨 예보 시 따뜻한 음료 프로모션을 진행하세요.")

        # 요일별 권장사항
        weekday_avg = data.groupby('day_of_week')['transaction_count'].mean()
        weekend_avg = data[data['is_weekend']]['transaction_count'].mean()
        weekday_sales_avg = data[~data['is_weekend']]['transaction_count'].mean()

        if weekend_avg > weekday_sales_avg * 1.1:
            recommendations.append("📅 주말 판매량이 평일보다 높습니다. 주말 특별 메뉴나 이벤트를 고려해보세요.")

        # 변동성 기반 권장사항
        cv = data['transaction_count'].std() / data['transaction_count'].mean()
        if cv > 0.3:
            recommendations.append("📊 일일 판매량 변동이 큽니다. 수요 예측 모델을 활용한 재고 관리가 필요합니다.")

        return recommendations