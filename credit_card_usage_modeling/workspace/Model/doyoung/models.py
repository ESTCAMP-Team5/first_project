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