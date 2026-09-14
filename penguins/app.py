import os
import requests
import streamlit as st

st.set_page_config(page_title='Palmer Penguins', page_icon='🐧')
st.title('🐧 Определение вида пингвина')
st.write('Введите четыре измерения. Модель определит вид: Adelie, Chinstrap или Gentoo.')
with st.form('measurements'):
    values = {
        'bill_length_mm': st.number_input('Длина клюва, мм', min_value=1.0, value=39.1),
        'bill_depth_mm': st.number_input('Глубина клюва, мм', min_value=1.0, value=18.7),
        'flipper_length_mm': st.number_input('Длина ласта, мм', min_value=1.0, value=181.0),
        'body_mass_g': st.number_input('Масса тела, г', min_value=1.0, value=3750.0),
    }
    submitted = st.form_submit_button('Определить вид')
if submitted:
    try:
        response = requests.post(os.getenv('API_URL', 'http://localhost:8000') + '/predict', json=values, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.success(f"Вид: {result['species']}")
        st.bar_chart(result['probabilities'])
    except (requests.RequestException, ValueError, KeyError):
        st.error('Не удалось получить предсказание. Попробуйте ещё раз после запуска API.')
