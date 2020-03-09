from loguru import logger
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import cm
import flirimageextractor
import io
from os import listdir, mkdir
from os.path import isfile, join
import os


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

multiple_images("/Volumes/Jaimyn USB/veolia/thermal")