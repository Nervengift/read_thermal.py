import matplotlib
import os
import subprocess
import numpy as np
from flir_image_extractor import FlirImageExtractor
from loguru import logger
from os import listdir, mkdir
from os.path import isfile, join

matplotlib.use("TkAgg")


def fix_metadata(folder_path: str, palette_names: list, original_images: list) -> None:
    """
    Copies metadata from original images to newly outputted images
    :param folder_path:
    :param palette_names:
    :param original_images:
    :return:
    @author Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    for image in original_images:
        # Build array of outputted images from the current original image
        processed_images = []
        for palette in palette_names:
            processed_image = "".join(image.split("/")[-1])[:-4] + f"_thermal_{palette}.JPG"
            if isfile(join(folder_path, palette, processed_image)) and not processed_image.startswith("."):
                processed_images.append({"palette": palette, "file_name": processed_image})

        # Copy metadata from original image to processed images
        for processed_image in processed_images:
            process_image_filename = join(folder_path, processed_image["palette"], processed_image["file_name"])
            subprocess.run(["exiftool", "-tagsfromfile", image, process_image_filename])

    subprocess.run(["exiftool", "-delete_original!", "-r", folder_path])


def multiple_images(folder_path: str, palettes: list, custom_temperature: dict) -> None:
    """
    Processes multiple images at the folder path and with the color map palettes provided
    :param folder_path:
    :param palettes:
    :param custom_temperature:
    :return:
    @author Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    flir = FlirImageExtractor(palettes=palettes)

    # get a list of all of the files in the folder and filter for just thermal images
    thermal_images = [
        join(folder_path, f)
        for f in listdir(folder_path)
        if isfile(join(folder_path, f)) and f.lower().endswith(".jpg") and not f.startswith(".") and flir.check_for_thermal_image(join(folder_path, f))
    ]

    # create a folder to store everything in
    if not os.path.exists(join(folder_path, "thermal-data")):
        mkdir(join(folder_path, "thermal-data"))

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
            folder_path,
            "thermal-data",
            "".join("".join(image.split("/")[-1]).split(".")[:-1]),
        )
        np.save(filename, thermal_data)

    # If custom temperatures were specified use those
    if custom_temperature["use_custom"]:
        dataset_min = custom_temperature["min"]
        dataset_max = custom_temperature["max"]

    # or using all the thermal data calculate the minimum and maximum temperatures for the dataset
    else:
        dataset_min = np.amin(min_values)
        dataset_max = np.amax(max_values)

    logger.log("SECTION", "Processing and saving the images to disk")
    # using these minTemp maxTemp values, save all of the images to disk
    for image in thermal_images:
        flir.process_image(image)
        flir.save_images(minTemp=dataset_min, maxTemp=dataset_max)
