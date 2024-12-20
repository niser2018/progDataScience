#!/usr/bin/env python
# coding: utf-8

# In[1]:


#графический интерфес
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
import PySimpleGUI as sg
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.utils import resample



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



    
    
    bootstrap_sample = resample(data_pd_del, replace=True, n_samples=len(data_pd_del), random_state=1)
    bootstrap_sample = resample(bootstrap_sample, replace=True, n_samples=len(bootstrap_sample), random_state=1)
    bootstrap_sample = resample(bootstrap_sample, replace=True, n_samples=len(bootstrap_sample), random_state=1)
    data_pd_del = bootstrap_sample.copy()
    
    minmax_customer_id = data_pd_del.copy()  
    minmax_customer_id = minmax_customer_id.drop(column, axis = 1)
    columns3 = minmax_customer_id.columns
    scaler3 = MinMaxScaler()
    minmax_customer_id[columns3] = scaler3.fit_transform(minmax_customer_id)
    
    
    
    if column == 'Соотношение матрица-наполнитель':
        mae = 0.000000000
        mse = 0.000000000
        rmse = 0.000000000
        mape = 0.000000000
        r2 = 0.000000000
        return mae, mse, rmse, mape, r2 
           
    
    if column == 'Модуль упругости при растяжении, ГПа':
        X = minmax_customer_id
        y = data_pd_del[column]
        y = y.values.ravel()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2, random_state=42)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        with open('mod_uprug.pkl', 'wb') as file:
            pickle.dump(model, file)
        
                
        
        
        # оцениваем модель
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / (np.array(y_test)+1e-10))) * 100
        r2 = r2_score(y_test, y_pred)
            
        return mae, mse, rmse, mape, r2
        
    if column == 'Прочность при растяжении, МПа':
        X = minmax_customer_id
        y = data_pd_del[column]
        y = y.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2, random_state=42)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        with open('mod_uprug.pkl', 'wb') as file:
            pickle.dump(model, file)
        
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
        window['-OUTPUT2-'].update(f'MSE: {mse:.3f}')
        window['-OUTPUT3-'].update(f'RMSE: {rmse:.3f}')
        window['-OUTPUT4-'].update(f'MAPE: {mape:.3f}')
        window['-OUTPUT5-'].update(f'R2: {r2:.3f}')
# закрываем окно и освобождаем используемые ресурсы
window.close()


# In[ ]:




