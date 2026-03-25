import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели, скейлера и списка признаков
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_top6.pkl')
    scaler = joblib.load('scaler_top6.pkl')
    features = joblib.load('top6_features.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

st.set_page_config(page_title="Калькулятор стадии глаукомы", layout="centered")
st.title("Калькулятор стадии глаукомы")
st.markdown("Введите 6 параметров для предсказания стадии (I, II или III)")

# Словарь для отображения имён признаков на русском
feature_labels = {
    'MD АСКП': 'MD (среднее отклонение), дБ',
    'Возраст': 'Возраст, лет',
    'PSD': 'PSD (паттерн), дБ',
    'sup 90°': 'sup 90° (порог в супериорном секторе), дБ',
    'СНВС 6 темп ниж': 'СНВС 6 темп ниж (толщина RNFL), мкм',
    'GCL 5 ниж': 'GCL 5 ниж (толщина ганглиозного слоя), мкм'
}

# Поля ввода
input_values = []
for f in feature_names:
    label = feature_labels.get(f, f)
    if 'Возраст' in f:
        val = st.number_input(label, min_value=18, max_value=100, value=65, step=1)
    elif 'MD' in f:
        val = st.number_input(label, min_value=-30.0, max_value=0.0, value=-3.0, step=0.5, format="%.1f")
    elif 'PSD' in f:
        val = st.number_input(label, min_value=0.0, max_value=15.0, value=2.5, step=0.5, format="%.1f")
    elif 'sup 90°' in f:
        val = st.number_input(label, min_value=0.0, max_value=50.0, value=40.0, step=1.0, format="%.1f")
    elif 'СНВС' in f or 'GCL' in f:
        val = st.number_input(label, min_value=10.0, max_value=150.0, value=70.0, step=1.0, format="%.1f")
    else:
        val = st.number_input(label, value=0.0)
    input_values.append(val)

# Кнопка предсказания
if st.button("Предсказать стадию"):
    # Преобразуем в массив и масштабируем
    X_input = np.array(input_values).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    stage_map = {1: "I (начальная)", 2: "II (развитая)", 3: "III (далекозашедшая)"}
    st.success(f"**Предсказанная стадия:** {stage_map.get(int(pred), 'неизвестно')}")
    # Дополнительно можно вывести вероятности
    probs = model.predict_proba(X_scaled)[0]
    st.write("**Вероятности:**")
    for i, prob in enumerate(probs, 1):
        st.write(f"- Стадия {i}: {prob:.2%}")