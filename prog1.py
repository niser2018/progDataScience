#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#графический интерфес
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
import PySimpleGUI as sg
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def load_dataframe(filepath):
    df = pd.read_excel(rf'{filepath}')
    return df

def update(data_pd, column):
    # функция обучения модели на основе выбранной колонки
    numeric_columns = data_pd.select_dtypes(include=['int64', 'float64']).columns
    data_pd_del = pd.DataFrame()
    for col in numeric_columns:
        Q1 = data_pd[col].quantile(0.25)
        Q3 = data_pd[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_pd_del = data_pd[(data_pd[col] > lower_bound) & (data_pd[col] < upper_bound)]
    scaler = Normalizer()
    normal_customer_id = data_pd_del.copy(deep=True)
    normal_customer_id[numeric_columns] = scaler.fit_transform(normal_customer_id)
 
    if column == 'Соотношение матрица-наполнитель':
        mae = 0.000000000
        mse = 0.000000000
        rmse = 0.000000000
        mape = 0.000000000
        r2 = 0.000000000
        return mae, mse, rmse, mape, r2 
           
    
    if column == 'Модуль упругости при растяжении, ГПа':
        X = normal_customer_id.drop(column, axis=1)
        y = normal_customer_id[column]
        y = y.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True, random_state=42)
      
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            
        }
        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                                   param_grid=param_grid, 
                                   cv=10,  # количество блоков для перекрестной проверки
                                   scoring='neg_mean_squared_error',  # метрика, которую хотим оптимизировать
                                   n_jobs=-1  # ядра процессора для более быстрого вычисления
                                  )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_    
        y_pred = best_model.predict(X_test)
        
        # оцениваем модель
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / (np.array(y_test)+1e-10))) * 100
        r2 = r2_score(y_test, y_pred)
            
        return mae, mse, rmse, mape, r2
        
    if column == 'Прочность при растяжении, МПа':
        X = normal_customer_id.drop(column, axis=1)
        y = normal_customer_id[column]
        y = y.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True, random_state=42)
      
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            
        }
        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                                   param_grid=param_grid, 
                                   cv=10,  # количество блоков для перекрестной проверки
                                   scoring='neg_mean_squared_error',  # метрика, которую хотим оптимизировать
                                   n_jobs=-1  # ядра процессора для более быстрого вычисления
                                  )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_    
        y_pred = best_model.predict(X_test)
        
        # оцениваем модель
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / (np.array(y_test)+1e-10))) * 100
        r2 = r2_score(y_test, y_pred)
        return mae, mse, rmse, mape, r2

    pass

layout = [[sg.Text('Обучение моделей для предсказаний:', size=(70, 3), font='Helvetica 16')],
          [sg.Input('путь до файла', key='-FILE-',), sg.FileBrowse()],
          [sg.Text('Выберите столбец для предсказания:')],
          [sg.Combo(values=[], key='-COLUMN-', size=(60, 1))],
          [sg.Button('Загрузить датасет'), sg.Button('Создать модель'), sg.Cancel()],
          [sg.Text(size=(60,1), key='-OUTPUT1-')],  # добавляем поле для вывода mse
          [sg.Text(size=(60,1), key='-OUTPUT2-')],
          [sg.Text(size=(60,1), key='-OUTPUT3-')],
          [sg.Text(size=(60,1), key='-OUTPUT4-')],
          [sg.Text(size=(60,1), key='-OUTPUT5-')],
          [sg.Text('Модель сформирована')]
         ]

window = sg.Window('Расчет параметров', layout, size=(550,400))

# запускаем основной бесконечный цикл
while True:
    # получаем события, произошедшие в окне
    event, values = window.read()
    # если нажали на крестик
    if event in (sg.WIN_CLOSED, 'Exit', 'Cancel'):
        # выходим из цикла
        break
    # если нажали на кнопку Load DataFrame
    if event == 'Загрузить датасет':
        df = load_dataframe(values['-FILE-'])
        col = [df.columns[i] for i in [0,7,8]]
        window['-COLUMN-'].update(values=col)
    # если нажали на кнопку Predict
    if event == 'Создать модель':
        mae, mse, rmse, mape, r2 = update(df, values['-COLUMN-'])  # получаем mse из функции update
        window['-OUTPUT1-'].update(f'MAE: {mae:.3f}')  # обновляем текстовое поле с ключом '-OUTPUT-'
        window['-OUTPUT2-'].update(f'MSE: {mse:.7f}')
        window['-OUTPUT3-'].update(f'RMSE: {rmse:.3f}')
        window['-OUTPUT4-'].update(f'MAPE: {mape:.3f}')
        window['-OUTPUT5-'].update(f'R2: {r2:.3f}')
# закрываем окно и освобождаем используемые ресурсы
window.close()

