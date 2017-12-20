# Usage

This tool relies on `exiftool`. It should be available in most Linux distributions (e.g. as `perl-image-exiftool` in Arch Linux or `libimage-exiftool-perl` in Debian and Ubuntu).

It also needs the Python packages *numpy* and *matplotlib* (the latter only if used interactively).

Calling it as `python read_thermal.py flir_example.jpg` will show an interactive plot of the thermal image using matplotlib and create two image files *flir_example_thermal.png* and *flir_example_visual.jpg*. Both are RGB images, the original temperature array is only available using the `extract_thermal` function.

The functions `extract_thermal` and `extract_visual` yield numpy arrays and can be called from your own script after importing this lib.

# Known bugs

I only tested this on photos taken with a FLIR ONE®.
Other cameras might need some small tweaks (the embedded raw data can be in multiple image formats, I only support png thermal and jpg visual for now)

# Credits

Raw value to temperature conversion is ported from this R package: https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R

