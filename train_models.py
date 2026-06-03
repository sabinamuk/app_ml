import pandas as pd
import pickle
import optuna
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import os
import pandas as pd

# Добавьте этот блок в начало train_models.py
if os.getenv('STREAMLIT_SERVER_PORT'): # Проверка, что мы в облаке
    # Читаем только первые 500-1000 строк для быстрой проверки
    df = pd.read_csv('data_new.csv', encoding='cp1251', nrows=1000) 
else:
    df = pd.read_csv('data_new.csv', encoding='cp1251')

df = pd.read_csv('data_new.csv')
X = df.drop('price', axis=1)
y = df['price']


def objective(trial):
    degree = trial.suggest_int('degree', 1, 3)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    return model.score(X, y)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_degree = study.best_params['degree']
ml1 = Pipeline([('poly', PolynomialFeatures(degree=best_degree)), ('scaler', StandardScaler()), ('linear', LinearRegression())])
ml1.fit(X, y)
with open('models/linear.pkl', 'wb') as f: pickle.dump(ml1, f)


ml2 = GradientBoostingRegressor().fit(X, y)
with open('models/gb.pkl', 'wb') as f: pickle.dump(ml2, f)


ml3 = lgb.LGBMRegressor().fit(X, y)
with open('models/lightgbm.pkl', 'wb') as f: pickle.dump(ml3, f)


ml4 = BaggingRegressor().fit(X, y)
with open('models/bagging.pkl', 'wb') as f: pickle.dump(ml4, f)

ml5 = StackingRegressor(estimators=[('gb', ml2), ('lgbm', ml3)], final_estimator=LinearRegression()).fit(X, y)
with open('models/stacking.pkl', 'wb') as f: pickle.dump(ml5, f)


scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
ml6 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
ml6.compile(optimizer='adam', loss='mse')
ml6.fit(X_scaled, y, epochs=20, verbose=0)
ml6.save('models/neural_net.keras')
with open('models/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
