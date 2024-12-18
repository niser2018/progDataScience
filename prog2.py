#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from sklearn.ensemble import GradientBoostingRegressor 
import pandas as pd
from sklearn.preprocessing import Normalizer
import pickle

def open_file():
    global data_pd
    filename = filedialog.askopenfilename()
    print(filename)  
    data_pd  = pd.read_excel(filename)

def select_option():
    df = data_pd.copy()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df['Угол нашивки, град'] = df['Угол нашивки, град'].replace([90, 0], [1, 0])
    data_pd_del = pd.DataFrame()
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_pd_del = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    col_pred = ['Модуль упругости при растяжении, ГПа']
    scaler = Normalizer()
    normal_customer_id = data_pd_del.copy(deep=True)
    normal_customer_id[numeric_columns] = scaler.fit_transform(normal_customer_id)
    X = normal_customer_id.drop(col_pred, axis=1)
    with open('mod_uprug.pkl', 'rb') as file:
        model = pickle.load(file)
      
    # делаем предсказания
    row_number = int(row_entry.get())
    result = model.predict(pd.DataFrame(X.iloc[row_number].values.reshape(1, -1), columns=X.columns))
    result_text.set(f"Расчитенный МОДУЛЬ УПРУГОСТИ равен: {result}")

    # получаем строку данных из исходного DataFrame
    data_pd_drop = data_pd.copy()
    data_pd_drop = data_pd_drop.drop(col_pred, axis=1)
    data_row = data_pd_drop.iloc[row_number]
    data_text.set(data_row.to_string())

root = tk.Tk()

# кнопка для загрузки файла
open_button = tk.Button(root, text="Выберите файл для проверки резултата", command=open_file)
open_button.pack()

# Текст действия
print_text = tk.StringVar()
print_text.set('Расчет модуля упругости')
print_label = tk.Label(root, textvariable=print_text)
print_label.pack()

# поле ввода для номера строки
def clear_entry(event, entry):
    entry.delete(0, tk.END)

row_entry = tk.Entry(root)
row_entry = tk.Entry(root, width=50)
row_entry.pack()
row_entry.insert(1, "Введите номер строки")
row_entry.bind("<FocusIn>", lambda event: clear_entry(event, row_entry))
# кнопка для выполнения кода
execute_button = tk.Button(root, text="Выполнить", command=select_option)
execute_button.pack()

# текстовое поле для вывода результата
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack()

data_text = tk.StringVar()
data_label = tk.Label(root, textvariable=data_text)
data_label.pack()

root.geometry("800x400") 

root.mainloop()

