import json
import os

import requests
import streamlit as st
from PIL import Image, ImageOps

from penguins.config import ASSETS, DISPLAY

REQUEST_TIMEOUT = 15
PHOTO_WIDTH = 300


def show_form(measurements):
    values = {}

    with st.form('measurements'):
        for name, settings in measurements.items():
            values[name] = st.number_input(
                settings['label'],
                min_value=0.1,
                value=settings['default'],
                help=f"В датасете: {settings['min']:g}–{settings['max']:g}",
            )

        submitted = st.form_submit_button('Определить вид')

    return values, submitted


def request_prediction(values):
    api_url = os.getenv('API_URL', 'http://localhost:8000')
    response = requests.post(
        api_url + '/predict',
        json=values,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def show_photo(species):
    image_path = ASSETS / species['image']

    if image_path.is_file():
        with Image.open(image_path) as photo:
            # Camera orientation is stored in EXIF, not in the pixel dimensions.
            upright_photo = ImageOps.exif_transpose(photo)
            st.image(upright_photo, caption=species['name'], width=PHOTO_WIDTH)
    else:
        st.info('Фотография вида недоступна.')

    st.caption(
        f"Фото: [{species['author']}]({species['source']}), "
        f"[{species['license']}]({species['license_url']})."
    )


def show_prediction(result, species):
    picture_column, result_column = st.columns([1, 1])

    with picture_column:
        show_photo(species)

    with result_column:
        st.success(
            f"Предсказанный вид: {species['name']} ({result['species']})"
        )
        st.bar_chart(result['probabilities'])
        st.caption(
            'Фотография иллюстрирует предсказанный вид. '
            'Модель использует только измерения.'
        )


def main():
    st.set_page_config(page_title='Palmer Penguins', page_icon='🐧')
    st.title('🐧 Определение вида пингвина')
    st.write('У вас есть пингвин, но вы не знаете, какого он вида? 🐧')
    st.write(
        'Укажите размеры клюва, длину ласта и массу — узнаем, кто перед нами: '
        'пингвин Адели, антарктический или папуанский!'
    )

    display = json.loads(DISPLAY.read_text(encoding='utf-8'))
    values, submitted = show_form(display['measurements'])
    if not submitted:
        return

    try:
        result = request_prediction(values)
        species = display['species'][result['species']]
        show_prediction(result, species)
    except (requests.RequestException, ValueError, KeyError):
        st.error(
            'Не удалось получить предсказание. '
            'Попробуйте ещё раз после запуска API.'
        )


if __name__ == '__main__':
    main()
