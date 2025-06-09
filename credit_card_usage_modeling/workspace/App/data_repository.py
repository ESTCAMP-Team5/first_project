import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import mysql.connector
from sqlalchemy import create_engine, text
import os
from typing import Optional
from dotenv import load_dotenv


class DataRepository(ABC):
    """데이터 접근을 위한 추상 클래스"""

    @abstractmethod
    def get_customer_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass


class MySQLDataRepository(DataRepository):
    def __init__(self):
        load_dotenv()
        # .env 파일 생성후 아래와 같이 기입
        # DB_HOST = localhost
        # DB_USER = root
        # DB_PASSWORD = 비밀번호 기입
        # DB_DATABASE = DB Name 기입

        self.host =  os.getenv("DB_HOST")
        self.port = 3306
        self.database = "suwon_business"
        self.username = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.engine = None
        self._create_engine()

    def _create_engine(self):
        """SQLAlchemy 엔진 생성"""
        try:
            connection_string = f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            # 연결 테스트 (SQLAlchemy 2.x에서는 text() 필요)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))   # ✅ 여기가 핵심 수정
            print("✅ MySQL 연결 성공!")
        except Exception as e:
            print(f"❌ MySQL 연결 실패: {str(e)}")
            raise

    def get_customer_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """MySQL에서 고객 거래 데이터 조회"""

        base_query = """
        SELECT
            t.transaction_date,
            t.district_code,
            d.name AS district_name,
            t.time_block_code,
            t.gender,
            t.age_group_code,
            t.day_of_week,
            t.transaction_amount,
            t.transaction_count,
            w.avg_temp
        FROM card_transaction t
        JOIN district d ON t.district_code = d.district_id
        JOIN weather w ON t.transaction_date = w.weather_date AND t.district_code = w.district_id
        """

        # 날짜 필터 추가
        conditions = []
        params = {}

        if start_date:
            conditions.append("t.transaction_date >= %(start_date)s")
            params['start_date'] = start_date

        if end_date:
            conditions.append("t.transaction_date <= %(end_date)s")
            params['end_date'] = end_date

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        # 정렬 추가
        base_query += " ORDER BY t.transaction_date, t.district_code"

        try:
            # pandas로 쿼리 실행
            df = pd.read_sql(base_query, self.engine, params=params)

            # 데이터 타입 정리
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['transaction_amount'] = df['transaction_amount'].astype(float)
            df['transaction_count'] = df['transaction_count'].astype(int)
            df['avg_temp'] = df['avg_temp'].astype(float)

            print(f"✅ 데이터 로드 완료: {len(df):,}건")
            return df

        except Exception as e:
            print(f"❌ 데이터 조회 실패: {str(e)}")
            raise

    def get_district_info(self) -> pd.DataFrame:
        """구 정보 조회"""
        query = """
        SELECT district_id, name AS district_name
        FROM district
        ORDER BY district_id
        """

        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"❌ 구 정보 조회 실패: {str(e)}")
            raise

    def get_date_range(self) -> dict:
        """데이터의 날짜 범위 조회"""
        query = """
        SELECT 
            MIN(transaction_date) as min_date,
            MAX(transaction_date) as max_date,
            COUNT(*) as total_records
        FROM card_transaction
        """

        try:
            result = pd.read_sql(query, self.engine)
            return {
                'min_date': result['min_date'].iloc[0],
                'max_date': result['max_date'].iloc[0],
                'total_records': result['total_records'].iloc[0]
            }
        except Exception as e:
            print(f"❌ 날짜 범위 조회 실패: {str(e)}")
            raise


class MockDataRepository(DataRepository):
    """테스트용 Mock Repository (MySQL 연결이 안될 때 사용)"""

    def __init__(self):
        self._generate_sample_data()
        print("⚠️ Mock 데이터 사용 중 (실제 MySQL 연결 필요)")

    def _generate_sample_data(self):
        """샘플 데이터 생성 (실제 DB 구조와 동일)"""
        np.random.seed(42)

        # 날짜 생성 (90일)
        start_date = datetime(2024, 8, 1)
        dates = [start_date + timedelta(days=i) for i in range(90)]

        # 온도 데이터 (계절성 반영)
        temperatures = 25 + 5 * np.sin(np.arange(90) * 2 * np.pi / 30) + np.random.normal(0, 2, 90)

        # 수원시 4개 구 정보 (실제 district_code 사용)
        districts = [
            (41111, "장안구"),
            (41113, "권선구"),
            (41115, "팔달구"),
            (41117, "영통구")
        ]

        data = []

        for i, date in enumerate(dates):
            # 하루에 20-40개 거래
            n_transactions = np.random.randint(20, 41)

            for _ in range(n_transactions):
                district_code, district_name = districts[np.random.randint(0, 4)]

                data.append({
                    'transaction_date': date,
                    'district_code': district_code,
                    'district_name': district_name,
                    'time_block_code': np.random.randint(1, 7),
                    'gender': np.random.choice(['F', 'M'], p=[0.6, 0.4]),
                    'age_group_code': np.random.randint(1, 7),
                    'day_of_week': date.weekday() + 1,
                    'transaction_amount': max(1000, int(np.random.lognormal(8.5, 0.8))),
                    'transaction_count': np.random.randint(1, 6),
                    'avg_temp': round(temperatures[i], 1)
                })

        self.sample_data = pd.DataFrame(data)

    def get_customer_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Mock 데이터 반환"""
        df = self.sample_data.copy()

        if start_date:
            df = df[df['transaction_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['transaction_date'] <= pd.to_datetime(end_date)]

        return df


def create_data_repository(repo_type: str = "mysql") -> DataRepository:
    if repo_type == "mysql":
        # 환경변수에서 연결 정보 가져오기 (없으면 kwargs 사용)
        try:
            return MySQLDataRepository()
        except Exception as e:
            print(f"⚠️ MySQL 연결 실패, Mock 데이터로 전환: {str(e)}")
            return MockDataRepository()

    elif repo_type == "mock":
        return MockDataRepository()

    else:
        raise ValueError(f"Unknown repository type: {repo_type}")
