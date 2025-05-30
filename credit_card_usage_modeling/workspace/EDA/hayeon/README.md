# 가설 설정

- 기온(X:AvgTemp)은 커피 매출(y:amt)에 있어서 상관관계가 있지 않다. (회귀분석)

import pandas as pd
import os

pip install pymysql

import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_DATABASE"),
    charset='utf8mb4'
)

query = """
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
    w.avg_temp,
    w.max_temp,
    w.min_temp
FROM card_transaction t
JOIN district d ON t.district_code = d.district_id
JOIN weather w ON t.transaction_date = w.weather_date AND t.district_code = w.district_id
"""

read_sql = pd.read_sql(query, conn)


from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+pymysql://root:1234@127.0.0.1:3306/suwon")

with engine.connect() as conn:
    for table in ['card_transaction', 'weather']:
        print(f"\n📄 {table} 테이블 구조:")
        df = pd.read_sql(f"SHOW COLUMNS FROM {table}", conn)
        print(df)

with engine.connect() as conn:
    df_card = pd.read_sql("SELECT * FROM card_transaction", conn)
    df_weather = pd.read_sql("SELECT * FROM weather", conn)

df_merged = pd.merge(df_card, df_weather, left_on='transaction_date', right_on='weather_date', how='left')



- 성별(X : sex)에 따라 매출(y:amt)에 영향이 있지 않다.(회귀분석)
- 커피 매장을 운영하면서 특정 외부 요인(X:모든 변수)에 의해 성별(y:sex)을 예측할 수 없다. (분류분석)
- 요일(X:day_of_week)에 따른 평균 커피 매출(y: amt) 은 영향이 없다.(회귀분석)
- 연령별 매출금액의 소비 수준(금액을 많이 쓰고 적게 쓰는 정도)패턴을 파악하기. (군집분석)
