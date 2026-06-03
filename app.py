import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import subprocess

# 1. Безопасный импорт TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Запуск обучения, если моделей нет
if not os.path.exists('models/linear.pkl'):
    st.warning("Модели не найдены. Запускаю процесс обучения...")
    subprocess.run(['python', 'train_models.py'], check=True)
    st.success("Обучение завершено!")
    st.rerun()

@st.cache_resource
def load_assets():
    
    models = {}
    model_names = ["linear", "rf", "gb", "bagging", "stacking"]
    for name in model_names:
        path = f'models/{name}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    nn = None
    if TF_AVAILABLE and os.path.exists('models/neural_net.keras'):
        try:
            nn = tf.keras.models.load_model('models/neural_net.keras')
        except:
            nn = None
            
    scaler = None
    if os.path.exists('models/scaler.pkl'):
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
    return models, nn, scaler

models, nn_model, scaler = load_assets()

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", ["Разработчик", "О данных", "Визуализация", "Предсказание"])

if page == "Разработчик":
    st.title("Информация о разработчике")
    st.write("**ФИО:** Муканова Сабина Толегеновна")
    st.write("**Группа:** ФИТ-242")
    st.write("**Тема РГР:** Дашборд предсказания стоимости недвижимости")

elif page == "О данных":
    st.title("Описание набора данных")
    st.write("Этот датасет содержит информацию о продажах домов в округе Кинг.")

elif page == "Визуализация":
    st.title("Анализ данных")
    if os.path.exists('data_new.csv'):
        df = pd.read_csv('data_new.csv', encoding='cp1251')
        st.subheader("Распределение цен")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax1)
        st.pyplot(fig1)
    else:
        st.error("Файл data_new.csv не найден.")

elif page == "Предсказание":
    st.title("Предсказание цены недвижимости")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("Спальни", value=3)
        bathrooms = st.number_input("Ванные", value=2)
    with col2:
        sqft_living = st.number_input("Площадь", value=1500)
        floors = st.number_input("Этаж", value=1)
    with col3:
        condition = st.slider("Состояние", 1, 5, 3)
        grade = st.slider("Качество", 1, 13, 7)
    
    age = st.number_input("Возраст дома", value=20)

    if st.button("Сделать прогноз"):
        feature_order = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                         'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                         'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15', 
                         'sqft_lot15', 'year_sale', 'age', 'age_renovated']
        
        input_data = pd.DataFrame([[
            bedrooms, bathrooms, sqft_living, 5000, floors, 0, 0, condition, 
            grade, 1000, 500, 32500, 47.5, -122.2, 1500, 5000, 2014, age, 0
        ]], columns=feature_order)
        
        st.write("### Результаты предсказаний:")
        
        if nn_model and scaler:
            input_scaled = scaler.transform(input_data)
            pred_nn = nn_model.predict(input_scaled, verbose=0)[0][0]
            st.write(f"**Нейросеть**: {pred_nn:,.2f} $")

        for name, model in models.items():
            pred = model.predict(input_data)[0]
            st.write(f"**{name.capitalize()}**: {pred:,.2f} $")
