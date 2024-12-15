#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
import PySimpleGUI as sg
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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
        epsilon = 1e-7
    scaler = MinMaxScaler()
    normalized_customer_id = data_pd_del.copy()
    normalized_customer_id[numeric_columns] = scaler.fit_transform(normalized_customer_id)
    normalized_customer_id
    if column == 'Соотношение матрица-наполнитель':
        X = normalized_customer_id.drop('Соотношение матрица-наполнитель', axis=1)
        y = normalized_customer_id['Соотношение матрица-наполнитель']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
            ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        model_save_path = r'my_model_2024.keras'
        checkpoint_callback_model = ModelCheckpoint(model_save_path, 
                                      monitor='val_mse',
                                      save_best_only=True,
                                      verbose=0)
        history = model.fit(X_train, 
          y_train, 
          epochs=30, 
          batch_size=32, 
          validation_data=(X_test, y_test),
          callbacks=[checkpoint_callback_model])
        mse = model.evaluate(X_test, y_test)[1]
        #mape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / (np.array(y_test)+epsilon))) * 100
        #r2 = r2_score(y_test, y_pred)
        return mse
        
    if column == 'Модуль упругости при растяжении, ГПа':
        X = normalized_customer_id.drop(['Модуль упругости при растяжении, ГПа'], axis=1)
        y = normalized_customer_id[['Модуль упругости при растяжении, ГПа']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
        param_grid = {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
        }
   
        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=10,  
                           scoring='neg_mean_squared_error',  
                           n_jobs=-1  
                          )
        grid_search.fit(X_train, y_train['Модуль упругости при растяжении, ГПа'])
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test['Модуль упругости при растяжении, ГПа'], y_pred)
        #rmse = mean_squared_error(y_test['Модуль упругости при растяжении, ГПа'], y_pred, squared=False)
        #mape = np.mean(np.abs((np.array(y_test['Модуль упругости при растяжении, ГПа']) - np.array(y_pred)) / (np.array(y_test['Модуль упругости при растяжении, ГПа'])+epsilon))) * 100
        #r2 = r2_score(y_test, y_pred)
        return mse
        
    if column == 'Прочность при растяжении, МПа':
        X = normalized_customer_id.drop(['Прочность при растяжении, МПа'], axis=1)
        y = normalized_customer_id[['Прочность при растяжении, МПа']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        param_grid = {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
        }
   
        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=10,  
                           scoring='neg_mean_squared_error',  
                           n_jobs=-1  
                          )
        grid_search.fit(X_train, y_train['Прочность при растяжении, МПа'])
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test['Прочность при растяжении, МПа'], y_pred)
        #rmse = mean_squared_error(y_test['Прочность при растяжении, МПа'], y_pred, squared=False)
        #mape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / (np.array(y_test)+epsilon))) * 100
        #r2 = r2_score(y_test, y_pred)
        return mse

    pass

layout = [[sg.Text('Обучение моделей для предсказаний:', size=(70, 3), font='Helvetica 16')],
          [sg.Input('путь до файла', key='-FILE-',), sg.FileBrowse()],
          [sg.Text('Выберите столбец для предсказания:')],
          [sg.Combo(values=[], key='-COLUMN-', size=(60, 1))],
          [sg.Button('Загрузить датасет'), sg.Button('Создать модель'), sg.Cancel()],
          [sg.Text(size=(60,1), key='-OUTPUT1-')],  # добавляем поле для вывода mse
          [sg.Text(size=(60,1), key='-OUTPUT2-')]
         ]

window = sg.Window('Расчет параметров', layout, size=(550,300))

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
        mse = update(df, values['-COLUMN-'])  # получаем mse из функции update
        window['-OUTPUT1-'].update(f'MSE: {mse}')  # обновляем текстовое поле с ключом '-OUTPUT-'
        #window['-OUTPUT2-'].update(f'RMSE: {rmse}')
# закрываем окно и освобождаем используемые ресурсы
window.close()


# In[ ]:




