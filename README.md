# ML Project - Предсказание справедливой цены подержанных смартфонов и ноутбуков

**Студент:** Елисеев Иван Дмитриевич  
**Группа:** БИВ233

## Описание задачи

**Задача:** Регрессия - предсказание рыночной цены б/у смартфона или ноутбука по его характеристикам.

**Данные:**
- [Used Phones & Tablets Pricing Dataset](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data) - 3454 объявления о б/у смартфонах
- [Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price) - 1303 ноутбука

Итого: **4661 объявление**, 27 признаков после feature engineering.

**Целевая метрика:** MAPE (Mean Absolute Percentage Error) - интуитивно понятна в контексте цен. Дополнительно: MAE, RMSE.

## Структура репозитория

```
.
├── data/
│   ├── raw/                    # исходные CSV (не в git)
│   └── processed/              # очищенные данные (не в git)
├── models/                     # сохранённые модели (не в git)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA: распределения, корреляции, выбросы
│   ├── 02_baseline.ipynb       # Ridge baseline без feature engineering
│   └── 03_experiments.ipynb    # RF, XGBoost, LightGBM, CatBoost, PCA, Stacking
├── src/
│   ├── preprocessing/
│   │   ├── clean.py            # очистка сырых данных, объединение датасетов
│   │   └── features.py         # feature engineering
│   └── models/
│       ├── evaluate.py         # метрики и train/val/test split
│       └── train.py            # сборка матрицы признаков, обучение финальной модели
├── tests/
│   └── test_pipeline.py        # 34 unit-теста
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Запуск

```bash
git clone <url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Скачать данные в `data/raw/`:
- `used_device_data.csv` с kaggle.com/datasets/ahsan81/used-handheld-device-data
- `laptop_price.csv` с kaggle.com/datasets/muhammetvarl/laptop-price

```bash
# Очистка и feature engineering
python3 -m src.preprocessing.clean
python3 -m src.preprocessing.features

# Обучение финальной модели
python3 -m src.models.train

# Тесты
python3 -m pytest tests/ -v
```

### Docker

```bash
docker-compose run preprocess   # очистка данных
docker-compose run train        # обучение модели
docker-compose run test         # тесты
```

## Результаты

| Модель | MAE (руб.) | RMSE (руб.) | MAPE | Набор |
|--------|-----------|------------|------|-------|
| Ridge Baseline | 18 961 | 26 592 | 42.3% | val |
| Ridge + PCA | 16 841 | 25 146 | 34.5% | val |
| RandomForest | 13 072 | 20 768 | 22.8% | val |
| XGBoost | 12 949 | 20 145 | 22.6% | val |
| CatBoost | 12 640 | 19 305 | 22.8% | val |
| Stacking (RF+LGB->Ridge) | 12 644 | 19 950 | 22.9% | val |
| LightGBM (num only) | 12 550 | 19 754 | 22.3% | val |
| **LightGBM (all features)** | **11 878** | **18 050** | **21.3%** | val |
| **LightGBM (all features)** | **11 747** | **18 191** | **19.6%** | **test** |

## Отчёт

[`report/report.md`](report/report.md)
