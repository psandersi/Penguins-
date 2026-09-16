import unittest
from unittest.mock import Mock, patch

import requests
from PIL import Image, ImageOps
from streamlit.testing.v1 import AppTest

from penguins.config import ASSETS, ROOT

APP_PATH = ROOT / 'penguins/deployment/app/web.py'


class AppTests(unittest.TestCase):
    def test_each_species_has_a_photo_without_warnings(self):
        for species in ['Adelie', 'Chinstrap', 'Gentoo']:
            with self.subTest(species=species):
                response = Mock()
                response.json.return_value = {
                    'species': species,
                    'probabilities': {'Adelie': 0.2, 'Chinstrap': 0.3, 'Gentoo': 0.5},
                    'warnings': ['Outside dataset range'],
                }
                with patch('requests.post', return_value=response):
                    page = AppTest.from_file(str(APP_PATH)).run()
                    self.assertEqual(len(page.number_input), 4)
                    page.button[0].click().run()

                self.assertFalse(page.exception)
                self.assertFalse(page.warning)
                self.assertIn(species, page.success[0].value)
                self.assertEqual(len(page.get('image')), 1)

    def test_api_unavailable_shows_error(self):
        with patch('requests.post', side_effect=requests.ConnectionError):
            page = AppTest.from_file(str(APP_PATH)).run()
            page.button[0].click().run()

        self.assertFalse(page.exception)
        self.assertEqual(len(page.error), 1)
        self.assertFalse(page.success)

    def test_chinstrap_orientation(self):
        with Image.open(ASSETS / 'chinstrap.jpg') as photo:
            upright_photo = ImageOps.exif_transpose(photo)
            self.assertGreater(upright_photo.height, upright_photo.width)


if __name__ == '__main__':
    unittest.main()
