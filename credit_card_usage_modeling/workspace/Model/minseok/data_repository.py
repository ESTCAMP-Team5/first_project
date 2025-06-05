import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import sqlite3
from abc import ABC, abstractmethod


class DataRepository(ABC):

    @abstractmethod
    def get_coffee_sales_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_temperature_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_combined_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass


class MockDataRepository(DataRepository):
    """샘플 데이터를 위한 Mock Repository"""

    def __init__(self):
        self._generate_sample_data()

    def _generate_sample_data(self):
        """샘플 데이터 생성"""
        np.random.seed(42)
        n_days = 90

        # 날짜 생성
        start_date = datetime(2024, 1, 1)
        self.dates = [start_date + timedelta(days=i) for i in range(n_days)]

        # 온도 데이터 (계절성 반영)
        day_of_year = np.array([d.timetuple().tm_yday for d in self.dates])
        temp_base = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        self.temperatures = temp_base + np.random.normal(0, 2, n_days)

        # 거래량 (온도와 상관관계)
        self.transactions = (self.temperatures * 150 + np.random.normal(0, 200, n_days) + 2000).astype(int)
        self.transactions = np.clip(self.transactions, 1000, 6000)

    def get_coffee_sales_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """커피 판매 데이터 조회"""
        df = pd.DataFrame({
            'date': self.dates,
            'transaction_count': self.transactions,
            'day_of_week': [d.weekday() for d in self.dates],
            'month': [d.month for d in self.dates],
            'is_weekend': [d.weekday() >= 5 for d in self.dates]
        })

        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df

    def get_temperature_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """온도 데이터 조회"""
        df = pd.DataFrame({
            'date': self.dates,
            'avg_temperature': np.round(self.temperatures, 1),
            'min_temperature': np.round(self.temperatures - np.random.uniform(2, 5, len(self.temperatures)), 1),
            'max_temperature': np.round(self.temperatures + np.random.uniform(2, 5, len(self.temperatures)), 1)
        })

        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df

    def get_combined_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """결합된 데이터 조회"""
        sales_data = self.get_coffee_sales_data(start_date, end_date)
        temp_data = self.get_temperature_data(start_date, end_date)

        combined_data = pd.merge(sales_data, temp_data, on='date', how='inner')
        return combined_data

    def get_summary_statistics(self) -> Dict:
        """요약 통계"""
        combined_data = self.get_combined_data()
        return {
            'total_days': len(combined_data),
            'avg_daily_transactions': combined_data['transaction_count'].mean(),
            'total_transactions': combined_data['transaction_count'].sum(),
            'avg_temperature': combined_data['avg_temperature'].mean(),
            'temperature_range': {
                'min': combined_data['avg_temperature'].min(),
                'max': combined_data['avg_temperature'].max()
            }
        }


class SQLiteDataRepository(DataRepository):
    """SQLite 데이터베이스를 위한 Repository (실제 DB 연결 예시)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """데이터베이스 초기화 (테스트용)"""
        # 실제 구현에서는 이 부분을 제거하고 기존 DB를 사용
        mock_repo = MockDataRepository()
        combined_data = mock_repo.get_combined_data()

        with sqlite3.connect(self.db_path) as conn:
            combined_data.to_sql('coffee_sales', conn, if_exists='replace', index=False)

    def get_coffee_sales_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """커피 판매 데이터 조회"""
        query = """
        SELECT date, transaction_count, day_of_week, month, is_weekend
        FROM coffee_sales
        """
        params = []

        if start_date or end_date:
            query += " WHERE "
            conditions = []
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            return df

    def get_temperature_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """온도 데이터 조회"""
        query = """
        SELECT date, avg_temperature, min_temperature, max_temperature
        FROM coffee_sales
        """
        params = []

        if start_date or end_date:
            query += " WHERE "
            conditions = []
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            return df

    def get_combined_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """결합된 데이터 조회"""
        query = "SELECT * FROM coffee_sales"
        params = []

        if start_date or end_date:
            query += " WHERE "
            conditions = []
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            return df


# Factory 패턴으로 Repository 생성
def create_data_repository(repo_type: str = "mock", **kwargs) -> DataRepository:
    """Repository 팩토리 함수"""
    if repo_type == "mock":
        return MockDataRepository()
    elif repo_type == "sqlite":
        return SQLiteDataRepository(kwargs.get('db_path', 'coffee_sales.db'))
    else:
        raise ValueError(f"Unknown repository type: {repo_type}")