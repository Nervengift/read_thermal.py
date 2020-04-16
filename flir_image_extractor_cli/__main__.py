import os
import matplotlib
import sys
import dialogs
import processing
from loguru import logger
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.shortcuts import message_dialog
from matplotlib import cm
from flir_image_extractor import FlirImageExtractor
from os import listdir
from os.path import isfile, join


matplotlib.use("TkAgg")


def main():
    path = dialogs.path_dialog()

    if path is None:
        sys.exit("Successfully exited.")

    # check what is at the path the user just provided
    flir = FlirImageExtractor()

    thermal_images = []

    if os.path.isdir(path):
        title = "Processing the provided filepath..."

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
            sys.exit("There are no radiometric images in the directory you provided.")
        else:
            text = f"The directory you provided contains {len(thermal_images)} radiometric images."

    elif os.path.isfile(path):
        path_is_thermal = flir.check_for_thermal_image(path)
        if path_is_thermal:
            text = "Success, the image you have provided contains thermal data."
        else:
            sys.exit("The file you have provided does not contain thermal data.")

    else:
        sys.exit("The path you have provided does not exist.")

    message_dialog(title="Input Path Confirmation", text=text, ok_text="Continue").run()

    results_array = dialogs.output_options_checklist()

    if results_array is None or len(results_array) == 0:
        sys.exit("Successfully exited.")

    output_csv = "csv" in results_array

    if output_csv:
        results_array.remove("csv")

    palettes = []

    # Add the palette object to the palette array
    for result in results_array:
        palettes.append(getattr(cm, result))

    metadata = False

    # only ask the following if the user has selected a colormap
    if not (len(results_array) == 1 and output_csv):
        metadata = dialogs.metadata_dialog()

        custom_temperature = {
            "use_custom": False,
            "min": None,
            "max": None
        }

        valid = False
        while not valid:
            custom_temperature["use_custom"] = dialogs.custom_temperature_dialog()

            if not custom_temperature["use_custom"]:
                break

            custom_temperature["min"] = dialogs.custom_temperature_input("minimum")
            if custom_temperature["min"] is None or custom_temperature["min"] == "":
                continue

            custom_temperature["max"] = dialogs.custom_temperature_input("maximum")
            if custom_temperature["max"] is None or custom_temperature["max"] == "":
                continue

            valid = True

    logger.info("Processing the images...")
    processing.multiple_images(folder_path=path, palettes=palettes, custom_temperature=custom_temperature)

    # If copy metadata to new images was selected
    if metadata:
        logger.info("Processing metadata...")
        processing.fix_metadata(folder_path=path, palette_names=results_array, original_images=thermal_images)


if __name__ == "__main__":
    main()
