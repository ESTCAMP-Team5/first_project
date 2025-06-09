# models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Dict
import plotly.graph_objects as go


class CustomerPurchaseAnalyzer:
    """고객 구매 패턴 분석을 위한 클래스"""

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        df_processed = df.copy()

        # 성별 원-핫 인코딩
        df_encoded = pd.get_dummies(df_processed, columns=['gender'], prefix='gender')

        # 날짜 처리
        df_encoded['transaction_date'] = pd.to_datetime(df_encoded['transaction_date'])

        # 고객 그룹별 정렬
        df_encoded = df_encoded.sort_values(['gender_F', 'gender_M', 'age_group_code', 'transaction_date'])

        # 고객별 피처 생성
        customer_cols = ['gender_F', 'gender_M', 'age_group_code']

        df_encoded['past_7day_amt'] = df_encoded.groupby(customer_cols)['transaction_amount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).sum()
        )

        df_encoded['past_30day_cnt'] = df_encoded.groupby(customer_cols)['transaction_count'].transform(
            lambda x: x.rolling(window=30, min_periods=1).sum()
        )

        df_encoded['mean_amt_per_visit'] = df_encoded['transaction_amount'] / (df_encoded['transaction_count'] + 1e-5)

        return df_encoded

    def fit_model(self, df: pd.DataFrame) -> Dict:
        """모델 학습"""
        df_processed = self.prepare_data(df)

        # 피처 선택
        feature_columns = [
            'gender_F', 'gender_M', 'age_group_code', 'time_block_code',
            'day_of_week', 'district_code', 'avg_temp',
            'past_7day_amt', 'past_30day_cnt', 'mean_amt_per_visit'
        ]

        # 존재하는 피처만 선택
        self.feature_names = [col for col in feature_columns if col in df_processed.columns]

        X = df_processed[self.feature_names]
        y = df_processed['transaction_amount']

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # XGBoost 모델 학습
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # 성능 평가
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")

        return {
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'explanation_power': r2 * 100
        }

    def predict_by_districts(self, conditions: Dict) -> Dict:
        """4개 구별 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        districts = {
            41111: "장안구",
            41113: "권선구",
            41115: "팔달구",
            41117: "영통구"
        }

        results = {}

        for district_code, district_name in districts.items():
            # 예측 데이터 준비
            pred_data = conditions.copy()
            pred_data['district_code'] = district_code

            # 기본값 설정
            if 'past_7day_amt' not in pred_data:
                pred_data['past_7day_amt'] = 50000
            if 'past_30day_cnt' not in pred_data:
                pred_data['past_30day_cnt'] = 15
            if 'mean_amt_per_visit' not in pred_data:
                pred_data['mean_amt_per_visit'] = 10000

            # DataFrame으로 변환
            pred_df = pd.DataFrame([pred_data])

            # 피처 순서 맞추기
            pred_df = pred_df.reindex(columns=self.feature_names, fill_value=0)

            # 예측
            prediction = self.model.predict(pred_df)[0]

            results[district_name] = {
                'district_code': district_code,
                'predicted_amount': max(0, prediction)  # 음수 방지
            }

        return results

    def create_temp_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """기온별 평균 예측 매출 분석"""
        if not self.is_fitted:
            print("❌ 모델이 학습되지 않음")
            return go.Figure()

        try:
            # 데이터 전처리
            df_processed = self.prepare_data(data)
            print(f"✅ 전처리된 데이터 크기: {len(df_processed)}")
            print(f"✅ 기온 데이터 범위: {df_processed['avg_temp'].min():.1f}°C ~ {df_processed['avg_temp'].max():.1f}°C")

            # 피처 준비
            X = df_processed[self.feature_names]
            print(f"✅ 피처 데이터 크기: {X.shape}")

            # 예측
            predictions = self.model.predict(X)
            print(f"✅ 예측 완료, 예측값 범위: {predictions.min():,.0f} ~ {predictions.max():,.0f}")

            # 기온을 구간별로 묶기 (2도씩 묶음)
            df_processed['predicted_amt'] = predictions
            df_processed['temp_range'] = (df_processed['avg_temp'] // 2) * 2  # 2도 단위로 그룹화

            # 기온 구간별 평균 예측 매출 계산
            mean_amt_by_temp = df_processed.groupby('temp_range')['predicted_amt'].mean()
            print(f"✅ 기온 구간별 그룹 수: {len(mean_amt_by_temp)}")

            if len(mean_amt_by_temp) == 0:
                print("❌ 기온별 그룹 데이터가 없음")
                return go.Figure()

            # 차트 생성
            fig = go.Figure()

            # 구간 라벨 생성 (예: "20-22°C")
            temp_labels = [f"{int(temp)}-{int(temp) + 2}°C" for temp in mean_amt_by_temp.index]

            fig.add_trace(go.Bar(
                x=temp_labels,
                y=mean_amt_by_temp.values,
                text=[f"{int(y):,}원" for y in mean_amt_by_temp.values],
                textposition='outside',
                marker_color='lightcoral',
                name='예측 매출'
            ))

            fig.update_layout(
                title='기온 구간별 평균 예측 매출 (2도 간격)',
                xaxis=dict(title='기온 구간 (℃)', tickangle=45),
                yaxis=dict(title='예측 매출 (원)'),
                showlegend=False,
                bargap=0.3,
                height=500,
                template='plotly_dark'
            )

            print(f"✅ 차트 생성 완료")
            return fig

        except Exception as e:
            print(f"❌ 기온 분석 오류: {str(e)}")
            print(f"❌ 데이터 컬럼: {data.columns.tolist()}")
            return go.Figure()

    def create_time_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """시간대별 평균 예측 매출 분석"""
        if not self.is_fitted:
            return go.Figure()

        # 데이터 전처리
        df_processed = self.prepare_data(data)

        # 피처 준비
        X = df_processed[self.feature_names]

        # 예측
        predictions = self.model.predict(X)

        # 시간대별 평균 예측 매출 계산
        df_processed['predicted_amt'] = predictions
        mean_amt_by_time = df_processed.groupby('time_block_code')['predicted_amt'].mean()

        # 차트 생성
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=mean_amt_by_time.index.astype(int),
            y=mean_amt_by_time.values,
            text=[f"{round(int(y), -2):,}원" for y in mean_amt_by_time.values],
            textposition='outside',
            marker_color='skyblue',
            name='예측 매출'
        ))

        fig.update_layout(
            title='시간대별 평균 예측 매출',
            xaxis=dict(
                title='시간대 (1~10)',
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(title='예측 매출'),
            showlegend=False,
            bargap=0.2,
            height=500,
            template='plotly_dark'
        )

        return fig

    def get_optimal_conditions(self, data: pd.DataFrame) -> Dict:
        """최적 조건 분석"""
        if not self.is_fitted:
            return {}

        try:
            # 데이터 전처리
            df_processed = self.prepare_data(data)

            # 피처 준비
            X = df_processed[self.feature_names]

            # 예측
            predictions = self.model.predict(X)
            df_processed['predicted_amt'] = predictions

            # 기온 구간별 분석 (2도 단위)
            df_processed['temp_range'] = (df_processed['avg_temp'] // 2) * 2
            temp_analysis = df_processed.groupby('temp_range')['predicted_amt'].mean()
            optimal_temp_range = temp_analysis.idxmax()
            max_temp_sales = temp_analysis.max()

            # 시간대별 분석
            time_analysis = df_processed.groupby('time_block_code')['predicted_amt'].mean()
            optimal_time = time_analysis.idxmax()
            max_time_sales = time_analysis.max()

            return {
                'optimal_temp': {
                    'temperature_range': f"{int(optimal_temp_range)}-{int(optimal_temp_range) + 2}°C",
                    'predicted_sales': max_temp_sales,
                    'message': f"기온이 {int(optimal_temp_range)}-{int(optimal_temp_range) + 2}°C일 때 예측 매출이 가장 높습니다: 약 {max_temp_sales:,.0f}원"
                },
                'optimal_time': {
                    'time_block': optimal_time,
                    'predicted_sales': max_time_sales,
                    'message': f"시간대 {optimal_time}에 매출이 {int(max_time_sales):,}원으로 가장 높습니다"
                }
            }
        except Exception as e:
            print(f"❌ 최적 조건 분석 오류: {str(e)}")
            return {}