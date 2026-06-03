import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import os
import subprocess

if not os.path.exists('models/linear.pkl'):
    st.warning("Модели не найдены. Запускаю процесс обучения...")
    subprocess.run(['python', 'train_models.py'], check=True)
    st.success("Обучение завершено!")


@st.cache_resource
def load_assets():
    models = {}
    model_names = ["linear", "rf", "gb", "bagging", "stacking"]
    for name in model_names:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    nn = tf.keras.models.load_model('models/neural_net.keras')
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
    st.write("""
    Этот датасет содержит информацию о продажах домов в округе Кинг.
    **Основные признаки:**
    Характеристики здания: количество спален, ванных, площадь, этаж, качество, состояние, возраст дома 
    
    **Предобработка:** Из набора были исключены ID сделки и дата продажи, так как они не влияют на рыночную стоимость.
    """)

elif page == "Визуализация":
    st.title("Анализ данных")
    
    try:
        df = pd.read_csv('data_new.csv', encoding='cp1251')
        
        st.subheader("1. Распределение цен")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(df['price'], kde=True, ax=ax1)
        st.pyplot(fig1)
        
        st.subheader("2. Цена vs Площадь")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x='sqft_living', y='price', ax=ax2, alpha=0.5)
        st.pyplot(fig2)
        
        st.subheader("3. Зависимость цены от спален")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='bedrooms', y='price', ax=ax3)
        st.pyplot(fig3)
        
        st.subheader("4. Матрица корреляций")
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)
        
    except Exception as e:
        st.error(f"Не удалось загрузить данные для визуализации: {e}")

elif page == "Предсказание":
    st.title("Предсказание цены недвижимости")
    

    model_names = {
    "Полиномиальная регрессия": "linear",
    "Gradient Boosting": "gb",
    "LightGBM": "lightgbm",
    "Bagging Regressor": "bagging",
    "Stacking Regressor": "stacking",
    "Нейросеть": "neural_net"
}

    st.subheader("Введите характеристики дома:")
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

    if st.button("Сделать прогноз по всем моделям"):

        feature_order = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                         'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                         'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15', 
                         'sqft_lot15', 'year_sale', 'age', 'age_renovated']
        

        input_data = pd.DataFrame([[
            bedrooms, bathrooms, sqft_living, 5000, floors, 0, 0, condition, 
            grade, 1000, 500, 32500, 47.5, -122.2, 1500, 5000, 2014, age, 0
        ]], columns=feature_order)
        
        st.write("### Результаты предсказаний:")
        

        for display_name, file_key in model_names.items():
            if file_key == "neural_net":
         
                pred = nn_model.predict(input_data, verbose=0)[0][0]
                st.write(f"**{display_name}**: {pred:,.2f} $")
            else:
                with open(f'models/{file_key}.pkl', 'rb') as f:
                    model = pickle.load(f)
                pred = model.predict(input_data)[0]
                st.write(f"**{display_name}**: {pred:,.2f} $")
