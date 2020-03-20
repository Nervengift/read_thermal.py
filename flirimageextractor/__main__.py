from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.shortcuts import input_dialog
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter
import time
from loguru import logger
import matplotlib
import numpy as np
from prompt_toolkit.shortcuts import message_dialog

matplotlib.use("TkAgg")
from matplotlib import cm
import flirimageextractor
import io
from os import listdir, mkdir
from os.path import isfile, join
import os
import subprocess
from os import listdir, mkdir
from os.path import isfile, join
import sys


def fixMetadata(FOLDER_PATH):
    print(FOLDER_PATH)

    # get a list of all of the files in the folder and filter for just thermal images
    thermal_images = [
        join(FOLDER_PATH, f)
        for f in listdir(FOLDER_PATH)
        if isfile(join(FOLDER_PATH, f)) and not f.startswith(".")
    ]

    print(thermal_images)

    for image in thermal_images:
        filename = join(
            FOLDER_PATH, "bwr", "".join(image.split("/")[-1])[:-4] + "_thermal_bwr.jpg"
        )
        print(filename)
        subprocess.run(["exiftool", "-tagsfromfile", image, filename])


def multiple_images(FOLDER_PATH):
    flir = flirimageextractor.FlirImageExtractor(is_debug=True, palettes=[cm.magma])
    # get a list of all of the files in the folder and filter for just thermal images
    thermal_images = [
        join(FOLDER_PATH, f)
        for f in listdir(FOLDER_PATH)
        if isfile(join(FOLDER_PATH, f))
        and f.lower().endswith(".jpg")
        and not f.startswith(".")
        and flir.check_for_thermal_image(join(FOLDER_PATH, f))
    ]

    # create a folder to store everything in
    if not os.path.exists(join(FOLDER_PATH, "thermal-data")):
        mkdir(join(FOLDER_PATH, "thermal-data"))

    # create two arrays to keep track of the minTemp and maxTemp temperatures
    min_values = []
    max_values = []

    logger.level("SECTION", no=38, color="<yellow>", icon="🎯")
    logger.log("SECTION", "Getting thermal information for all images")
    # for each of the thermal images, get the thermal data (and save it to disk?)
    for image in thermal_images:
        flir.process_image(image)
        thermal_data = flir.get_thermal_np()

        # process the temperature data from the image
        max_values.append(np.amax(thermal_data))
        min_values.append(np.amin(thermal_data))

        filename = join(
            FOLDER_PATH,
            "thermal-data",
            "".join("".join(image.split("/")[-1]).split(".")[:-1]),
        )
        logger.info(f"Saving thermal data to {filename}.np")
        np.save(filename, thermal_data)

    # using all the thermal data calculate the minimum and maximum temperatures for the dataset
    dataset_min = np.amin(min_values)
    dataset_max = np.amax(max_values)

    logger.log("SECTION", "Saving the images to disk")
    # using these minTemp maxTemp values, save all of the images to disk
    for image in thermal_images:
        flir.process_image(image)
        flir.save_images(minTemp=dataset_min, maxTemp=dataset_max)


path = input_dialog(
    title="File or Directory Path", text="Input file or directory path: ",
).run()

# check what is at the path the user just provided
flir = flirimageextractor.FlirImageExtractor()

if os.path.isdir(path):
    title = HTML("Processing the provided filepath...")

    thermal_images = []
    with ProgressBar(title=title) as pb:
        for f in pb(listdir(path)):
            if (
                isfile(join(path, f))
                and f.lower().endswith(".jpg")
                and not f.startswith(".")
                and flir.check_for_thermal_image(join(path, f))
            ):
                thermal_images.append(join(path, f))

    if len(thermal_images) == 0:
        sys.exit("There are no radiometric images in the directory you provided :(")
    else:
        text = f"The directory you provided contains {len(thermal_images)} radiometric images"

elif os.path.isfile(path):
    path_is_thermal = flir.check_for_thermal_image(path)
    if path_is_thermal:
        text = "Success, the image you have provided contains thermal data"
    else:
        sys.exit("The file you have provided does not contain thermal data")

else:
    sys.exit("The path you have provided does not exist")

message_dialog(title="Input Path Confirmation", text=text, ok_text="Continue").run()

results_array = checkboxlist_dialog(
    title="CheckboxList dialog",
    text="Select one or more output options (press tab to select ok/cancel)",
    values=[
        ("csv", "CSV file containing temperature data (in degrees celcius)"),
        ("bwr", "Thermal image using bwr colormap"),
        ("gnuplot2", "Thermal image gnuplot2 colormap"),
        ("gist_ncar", "Thermal image gist_ncar colormap"),
        # ("custom", "Thermal image using a custom colormap"),
    ],
).run()

# TODO; implement custom colormaps
# if "custom" in results_array:
#     custom_colormap = input_dialog(
#         title="Custom Matplotlib Colormap",
#         text="Matplotlib colormap name: ",
#     ).run()

# only ask the following if the user has selected a colormap
if not (len(results_array) == 1 and results_array[0] == "csv"):
    metadata = yes_no_dialog(
        title="Metadata",
        text="Do you want all of the metadata from the original images copied to the new ones?",
    ).run()

    custom_min_max = yes_no_dialog(
        title="Custom Min and Max",
        text="Do you want to set custom min and max values?",
    ).run()

    if custom_min_max:
        custom_min = input_dialog(
            title="Custom Min",
            text="Input custom minimum temperature value (in degrees celcius): ",
        ).run()

        custom_max = input_dialog(
            title="Custom Max",
            text="Input custom maximum temperature value (in degrees celcius): ",
        ).run()

# actually do all the things that have been selected
title = HTML("Processing the images...")
with ProgressBar(title=title) as pb:
    for i in pb(range(800)):
        time.sleep(0.01)
