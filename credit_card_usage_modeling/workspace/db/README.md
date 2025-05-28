### ERD

```mermaid
erDiagram
    REGIONS {
        INT id PK
        VARCHAR sigungu_code
        VARCHAR adm_code
        VARCHAR sigungu_name
    }

    INDUSTRY_CATEGORIES {
        INT id PK
        VARCHAR category_code
        VARCHAR main_category
        VARCHAR sub_category
    }

    AGGREGATED_CONSUMPTION {
        INT id PK
        DATE ymd
        INT region_id FK
        INT industry_id FK
        INT hour
        ENUM sex
        INT age
        INT day_of_week
        BIGINT amt
        INT cnt
    }

    REGIONS ||--o{ AGGREGATED_CONSUMPTION : has
    INDUSTRY_CATEGORIES ||--o{ AGGREGATED_CONSUMPTION : has
```
