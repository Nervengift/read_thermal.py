from pathlib import Path

import numpy as np

from .thermal_extractor import ThermalExtractor


def test_process_image_works():
    image_path = Path(__file__).parent.parent / "examples/ax8.jpg"
    thermal = ThermalExtractor().process_image(image_path)
    np.testing.assert_array_less(thermal, 30)
