import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Оценка цены б/у устройства")
st.caption("Предсказание рыночной цены подержанного смартфона или ноутбука.")

device_type = st.radio("Тип устройства", ["smartphone", "laptop"], horizontal=True)

col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Бренд", value="Apple" if device_type == "smartphone" else "Dell")
    screen_inch = st.number_input("Диагональ экрана (дюймы)", min_value=3.0, max_value=20.0, value=6.1)
    storage_gb = st.number_input("Хранилище (ГБ)", min_value=8, max_value=4096, value=128)
    ram_gb = st.number_input("RAM (ГБ)", min_value=1, max_value=128, value=8)

with col2:
    os = st.text_input("ОС", value="iOS" if device_type == "smartphone" else "Windows")
    if device_type == "smartphone":
        condition = st.selectbox("Состояние", ["excellent", "good", "fair", "poor"])
        days_used = st.number_input("Дней в использовании", min_value=0, max_value=3000, value=365)
        rear_camera_mp = st.number_input("Основная камера (МП)", min_value=0, max_value=200, value=12)
        front_camera_mp = st.number_input("Фронтальная камера (МП)", min_value=0, max_value=100, value=12)
        battery_mah = st.number_input("Батарея (мАч)", min_value=0, max_value=10000, value=3227)
        release_year = st.number_input("Год выпуска", min_value=2000, max_value=2025, value=2021)
    else:
        cpu_brand = st.selectbox("Процессор", ["Intel", "AMD", "Other"])
        gpu_brand = st.selectbox("Видеокарта", ["Nvidia", "AMD", "Intel", "Other"])
        laptop_type = st.selectbox("Тип", ["Notebook", "Gaming", "Ultrabook", "Workstation", "Netbook", "2 in 1 Convertible"])
        storage_type = st.selectbox("Тип хранилища", ["SSD", "HDD"])
        weight_kg = st.number_input("Вес (кг)", min_value=0.5, max_value=10.0, value=1.8)

if st.button("Рассчитать цену", type="primary"):
    payload = {
        "device_type": device_type,
        "brand": brand,
        "os": os,
        "screen_inch": screen_inch,
        "storage_gb": storage_gb,
        "ram_gb": ram_gb,
    }
    if device_type == "smartphone":
        payload.update({
            "condition": condition,
            "days_used": days_used,
            "rear_camera_mp": rear_camera_mp,
            "front_camera_mp": front_camera_mp,
            "battery_mah": battery_mah,
            "release_year": release_year,
        })
    else:
        payload.update({
            "cpu_brand": cpu_brand,
            "gpu_brand": gpu_brand,
            "laptop_type": laptop_type,
            "storage_type": storage_type,
            "weight_kg": weight_kg,
        })

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        st.success(f"Оценочная цена: **{result['price_rub_formatted']}**")
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
