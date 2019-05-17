# Flir Image Extractor

FLIR® thermal cameras like the FLIR ONE® include both a thermal and a visual light camera.
The latter is used to enhance the thermal image using an edge detector.

The resulting image is saved as a jpg image but both the original visual image and the raw thermal sensor data are embedded in the jpg metadata.

This small Python tool/library allows to extract the original photo and thermal sensor values converted to temperatures.

## Requirements and Install

This tool relies on `exiftool`. It should be available in most Linux distributions (e.g. as `perl-image-exiftool` in Arch Linux or `libimage-exiftool-perl` in Debian and Ubuntu). Links for downloading the Mac version and more information is available on the [ExifTool site](https://sno.phy.queensu.ca/~phil/exiftool/index.html).

It also requires other python packages, *matplotlib*, *numpy* and *pillow*, which are installed when installed through pip.

```bash
sudo apt update
sudo apt install exiftool
pip install flirimageextractor
```

## Usage

#### Example

```python
import flirimageextractor
from matplotlib import cm

flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
flir.process_image('examples/ax8.jpg')
flir.save_images()
flir.plot()
```
This script will show an interactive plot of the thermal image using matplotlib and save three image files *ax8_thermal_jet.jpg*, *ax8_thermal_bwr.jpg*, and *ax8_thermal_gist_ncar.jpg*. 

#### Resulting Plot and Saved Images
##### Plot
![Python 2019-05-17 at 14 14 57](https://user-images.githubusercontent.com/8899750/57902766-2dd3ab00-78ae-11e9-9aba-bf033e481a34.png)

##### Saved Images
![ax8_thermal_jet](https://user-images.githubusercontent.com/8899750/57902729-0977ce80-78ae-11e9-9e7f-39800ffb7458.jpg)
![ax8_thermal_bwr](https://user-images.githubusercontent.com/8899750/57902822-7ab78180-78ae-11e9-9aac-f4b318b086b4.jpg)
![ax8_thermal_gist_ncar](https://user-images.githubusercontent.com/8899750/57902823-7be8ae80-78ae-11e9-8d50-20b1f1cc7818.jpg)

The original temperature array is available using either the `get_thermal_np` or `export_thermal_to_csv` functions.

The functions `get_rgb_np` and `get_thermal_np` yield numpy arrays and can be called from your own script after importing this library.

The function `save_image` saves the thermal image(s) in the same folder as the original image. By default it will output three images using the `bwr`, `gnuplot`, and `gist_ncar` colormaps from matplotlib. You can define the pallete(s) that you would rather use when creating the class (see example). For a list of available matplotlib colormaps click [here](https://matplotlib.org/tutorials/colors/colormaps.html)

## Supported/Tested cameras:

- Flir One (thermal + RGB)
- Xenmuse XTR (thermal + thumbnail, set the subject distance to 1 meter)
- AX8 (thermal + RGB)

Other cameras might need some small tweaks (the embedded raw data can be in multiple image formats)

## Credits

Raw value to temperature conversion is ported from this R package: https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
Original Python code from: https://github.com/Nervengift/read_thermal.py
