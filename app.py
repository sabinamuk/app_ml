import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка меню навигации
st.sidebar.title("Меню")
page = st.sidebar.radio("Навигация:", ["Разработчик", "Описание данных", "Визуализация", "Прогноз"])

# 1. Страница: Разработчик
if page == "Разработчик":
    st.title("Информация о разработчике")
    st.write("**ФИО:** Муканова Сабина Толегеновна")
    st.write("**Группа:** ФИТ-242")
    st.write("**Тема РГР:** Дашборд предсказания стоимости недвижимости")
    if os.path.exists("photo.jpg"):
        st.image("photo.jpg", width=200)

# 2. Страница: Описание данных
elif page == "Описание данных":
    st.title("Описание набора данных")
    st.write("""
    Этот датасет содержит информацию о продажах домов в округе Кинг.
    **Основные признаки:**
    - Характеристики здания: количество спален, ванных, площадь.
    - Географические данные: широта, долгота, почтовый индекс.
    - Особенности: состояние дома, класс качества, наличие набережной.
    
    **Предобработка:** Из набора были исключены ID сделки и дата продажи, так как они не влияют на рыночную стоимость.
    """)
# 3. Страница: Визуализация
elif page == "Визуализация":
    st.title("Анализ данных (EDA)")
    if os.path.exists('data_new.csv'):
        df = pd.read_csv('data_new.csv', encoding='cp1251')
        fig1, ax1 = plt.subplots(); sns.histplot(df['price'], ax=ax1); st.pyplot(fig1)
        fig2, ax2 = plt.subplots(); sns.scatterplot(data=df, x='sqft_living', y='price', ax=ax2); st.pyplot(fig2)
        fig3, ax3 = plt.subplots(); sns.boxplot(data=df, x='bedrooms', y='price', ax=ax3); st.pyplot(fig3)
        fig4, ax4 = plt.subplots(); sns.heatmap(df.corr(), annot=False, ax=ax4); st.pyplot(fig4)
    else:
        st.error("Файл данных не найден.")

# 4. Страница: Прогноз
elif page == "Прогноз":
    st.title("Дашборд предсказания цены на недвижимость")

    @st.cache_resource
    def load_models():
        import joblib
        models = {}
        # Загрузка классики
        models['Poly'] = joblib.load('models/model_ml1.pkl')
        models['GB'] = joblib.load('models/model_ml2.pkl')
        models['LGBM'] = lgb.Booster(model_file='models/model_ml3.txt')
        models['Bagging'] = joblib.load('models/model_ml4.pkl')
        models['Stacking'] = joblib.load('models/model_ml5.pkl')
        
        try:
            import onnxruntime as ort
            models['NeuralNet'] = ort.InferenceSession('models/model_ml6.onnx')
        except Exception as e:
            st.error(f"Ошибка загрузки нейросети (ONNX): {e}")
            models['NeuralNet'] = None
        return models

    if os.path.exists('models'):
        models = load_models() 
        
        st.header("Введите параметры дома")
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                bedrooms = st.number_input("Спальни", value=3)
                bathrooms = st.number_input("Ванные", value=2)
                sqft_living = st.number_input("Жилая площадь", value=2000)
                sqft_lot = st.number_input("Площадь участка", value=5000)
                floors = st.number_input("Этажи", value=1)
                waterfront = st.selectbox("Набережная (0/1)", [0, 1])
            with col2:
                view = st.slider("Вид (0-4)", 0, 4, 0)
                condition = st.slider("Состояние (1-5)", 1, 5, 3)
                grade = st.slider("Класс (1-13)", 1, 13, 7)
                sqft_above = st.number_input("Площадь над землей", value=1500)
                sqft_basement = st.number_input("Площадь подвала", value=500)
                zipcode = st.number_input("Индекс (zipcode)", value=98000)
            with col3:
                lat = st.number_input("Широта (lat)", value=47.5, format="%.4f")
                long = st.number_input("Долгота (long)", value=-122.0, format="%.4f")
                sqft_living15 = st.number_input("Жилая площадь 15 соседей", value=2000)
                sqft_lot15 = st.number_input("Площадь участка 15 соседей", value=5000)
                year_sale = st.number_input("Год продажи", value=2024)
                age = st.number_input("Возраст дома", value=20)
                age_renovated = st.number_input("Возраст после ремонта", value=0)
            
            submitted = st.form_submit_button("Рассчитать стоимость")

        if submitted:
            input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
                                        condition, grade, sqft_above, sqft_basement, zipcode, lat, long, 
                                        sqft_living15, sqft_lot15, year_sale, age, age_renovated]],
                                        columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                                                'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'zipcode', 
                                                'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sale', 'age', 'age_renovated'])
            
            st.subheader("Результаты предсказаний:")
            st.write(f"**Полиномиальная регрессия:** {models['Poly'].predict(input_data)[0]:,.2f} $")
            st.write(f"**Gradient Boosting:** {models['GB'].predict(input_data)[0]:,.2f} $")
            st.write(f"**LightGBM:** {models['LGBM'].predict(input_data)[0]:,.2f} $")
            st.write(f"**Bagging:** {models['Bagging'].predict(input_data)[0]:,.2f} $")
            st.write(f"**Stacking:** {models['Stacking'].predict(input_data)[0]:,.2f} $")
            
            if models['NeuralNet'] is not None:
                import numpy as np
                # Получаем имя входного слоя нейросети
                input_name = models['NeuralNet'].get_inputs()[0].name
                # Конвертируем данные в float32 (требование ONNX)
                input_array = input_data.values.astype(np.float32)
                # Делаем предсказание
                nn_pred = models['NeuralNet'].run(None, {input_name: input_array})[0]
                st.write(f"**Нейронная сеть:** {float(nn_pred[0][0]):,.2f} $")
    else:
        st.error("Папка 'models' не найдена.")
