import io
import json
import os
import re
import subprocess
from math import sqrt, exp

import numpy as np
from PIL import Image, ImageEnhance
from loguru import logger
from matplotlib import pyplot as plt, cm


class FlirImageExtractor:
    """
    Instance of FlirImageExtractor

    """

    def __init__(self, exiftool_path="exiftool", is_debug=False, palettes=None):
        if palettes is None:
            palettes = [cm.bwr, cm.gnuplot2, cm.gist_ncar]
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = None
        self.flir_img_bytes = None
        self.default_distance = 1.0
        self.rgb_image_np = None
        self.thermal_image_np = None
        self.palettes = palettes

        # valid for PNG thermal images
        self.fix_endian = True
        self.use_thumbnail = False

    def loadfile(self, file):
        """
        Loads an image file from a file path or a file-like object

        :param file: File path or file like object to load the image from
        :return:
        """
        if not isinstance(file, io.IOBase):
            if not os.path.isfile(file):
                raise ValueError(
                    "Input file does not exist or this user don't have permission on this file"
                )
            if self.is_debug:
                logger.debug("Flir image filepath:{}".format(file))

            self.flir_img_filename = file
        else:
            if self.is_debug:
                logger.debug("Loaded file from object")
            self.flir_img_bytes = file

    def get_metadata(self, flir_img_file):
        """
        Given a valid file path or file-like object get relevant metadata out of the image using exiftool.

        :param flir_img_file: File path or file like object to load the image from
        :return:
        """
        self.loadfile(flir_img_file)

        if self.flir_img_filename:
            meta_json = subprocess.check_output(
                [self.exiftool_path, self.flir_img_filename, "-j"]
            )
        else:
            args = ["exiftool", "-j", "-"]
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            meta_json, err = p.communicate(input=self.flir_img_bytes.read())

        return json.loads(meta_json.decode())[0]

    def check_for_thermal_image(self, flir_img_filename):
        """
        Given a valid image path, return a boolean of whether the image contains thermal data.

        :param flir_img_filename: File path or file like object to load the image from
        :return: Bool
        """
        metadata = self.get_metadata(flir_img_filename)
        return "RawThermalImageType" in metadata

    def process_image(self, flir_img_file, RGB=False):
        """
        Given a valid image path, process the file: extract real thermal values
        and an RGB image if specified

        :param flir_img_file: File path or file like object to load the image from
        :param RGB: Boolean for whether to extract the embedded RGB image
        :return:
        """
        # if bytesIO then save the image file to the class variable
        self.loadfile(flir_img_file)

        # if its a TIFF different settings are required
        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        # extract the thermal image and set it to the class variable
        self.thermal_image_np = self.extract_thermal_image()

        if RGB:
            self.rgb_image_np = self.extract_embedded_image()

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG

        :return:
        """
        if self.flir_img_filename:
            meta_json = subprocess.check_output(
                [
                    self.exiftool_path,
                    "-RawThermalImageType",
                    "-j",
                    self.flir_img_filename,
                ]
            )
        else:
            self.flir_img_bytes.seek(0)
            args = ["exiftool", "-RawThermalImageType", "-j", "-"]
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            meta_json, err = p.communicate(input=self.flir_img_bytes.read())

        return json.loads(meta_json.decode())[0]["RawThermalImageType"]

    def get_rgb_np(self):
        """
        Return the last extracted rgb image

        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image

        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values

        :return: Numpy Array of RGB values
        """
        image_tag = "-EmbeddedImage"

        if self.flir_img_filename:
            visual_img_bytes = subprocess.check_output(
                [self.exiftool_path, image_tag, "-b", self.flir_img_filename]
            )
        else:
            self.flir_img_bytes.seek(0)
            args = ["exiftool", image_tag, "-b", "-"]
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            visual_img_bytes, err = p.communicate(input=self.flir_img_bytes.read())

        visual_img_stream = io.BytesIO(visual_img_bytes)
        visual_img_stream.seek(0)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC

        :return: Numpy Array of thermal values
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        if self.flir_img_filename:
            meta_json = subprocess.check_output(
                [
                    self.exiftool_path,
                    self.flir_img_filename,
                    "-Emissivity",
                    "-SubjectDistance",
                    "-AtmosphericTemperature",
                    "-ReflectedApparentTemperature",
                    "-IRWindowTemperature",
                    "-IRWindowTransmission",
                    "-RelativeHumidity",
                    "-PlanckR1",
                    "-PlanckB",
                    "-PlanckF",
                    "-PlanckO",
                    "-PlanckR2",
                    "-j",
                ]
            )
        else:
            self.flir_img_bytes.seek(0)
            args = [
                "exiftool",
                "-Emissivity",
                "-SubjectDistance",
                "-AtmosphericTemperature",
                "-ReflectedApparentTemperature",
                "-IRWindowTemperature",
                "-IRWindowTransmission",
                "-RelativeHumidity",
                "-PlanckR1",
                "-PlanckB",
                "-PlanckF",
                "-PlanckO",
                "-PlanckR2",
                "-j",
                "-",
            ]
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            meta_json, err = p.communicate(input=self.flir_img_bytes.read())

        meta = json.loads(meta_json.decode())[0]

        # use exiftool to extract the thermal images
        if self.flir_img_filename:
            thermal_img_bytes = subprocess.check_output(
                [self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename]
            )
        else:
            self.flir_img_bytes.seek(0)
            args = ["exiftool", "-RawThermalImage", "-b", "-"]
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            thermal_img_bytes, err = p.communicate(input=self.flir_img_bytes.read())

        thermal_img_stream = io.BytesIO(thermal_img_bytes)
        thermal_img_stream.seek(0)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if "SubjectDistance" in meta:
            subject_distance = FlirImageExtractor.extract_float(meta["SubjectDistance"])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.right_shift(thermal_np, 8) + np.left_shift(
                np.bitwise_and(thermal_np, 0x00FF), 8
            )

        # run the thermal data numpy array through the raw2temp conversion
        return FlirImageExtractor.raw2temp(
            thermal_np,
            E=meta["Emissivity"],
            OD=subject_distance,
            RTemp=FlirImageExtractor.extract_float(
                meta["ReflectedApparentTemperature"]
            ),
            ATemp=FlirImageExtractor.extract_float(meta["AtmosphericTemperature"]),
            IRWTemp=FlirImageExtractor.extract_float(meta["IRWindowTemperature"]),
            IRT=meta["IRWindowTransmission"],
            RH=FlirImageExtractor.extract_float(meta["RelativeHumidity"]),
            PR1=meta["PlanckR1"],
            PB=meta["PlanckB"],
            PF=meta["PlanckF"],
            PO=meta["PlanckO"],
            PR2=meta["PlanckR2"],
        )

    @staticmethod
    def raw2temp(
            raw,
            E=1,
            OD=1,
            RTemp=20,
            ATemp=20,
            IRWTemp=20,
            IRT=1,
            RH=50,
            PR1=21106.77,
            PB=1501,
            PF=1,
            PO=-7340,
            PR2=0.012545258,
    ):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(
            1.5587
            + 0.06939 * (ATemp)
            - 0.00027816 * (ATemp) ** 2
            + 0.00000068455 * (ATemp) ** 3
        )
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
        )
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
        )

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

        raw_obj = (
                raw / E / tau1 / IRT / tau2
                - raw_atm1_attn
                - raw_atm2_attn
                - raw_wind_attn
                - raw_refl1_attn
                - raw_refl2_attn
        )

        # temperature from radiance
        temp_celcius = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirty_str):
        """
        Extract the float value of a string, helpful for parsing the exiftool data.

        :param dirty_str: The string to parse the float from
        :return: The float parsed from the string
        """

        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirty_str)
        return float(digits[0])

    def plot(self, palette=cm.gnuplot2):
        """
        Plot the rgb and thermal image (easy to see the pixel values), include a matplotlib colormap to change the colors

        :param palette: A matplotlib colormap to display the thermal image in
        :return:
        """
        plt.subplot(1, 2, 1)
        plt.imshow(self.thermal_image_np, cmap=palette)

        if self.rgb_image_np is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(self.rgb_image_np)

        plt.show()

    def save_images(self, minTemp=None, maxTemp=None, bytesIO=False):
        """
        Save the extracted images

        :param minTemp: (Optional) Manually set the minimum temperature for the colormap to use
        :param maxTemp: (Optional) Manually set the maximum temperature for the colormap to use
        :param bytesIO: (Optional) Return an array of BytesIO objects containing the images rather than saving to disk
        :return: Either a list of filenames where the images were save, or an array containing BytesIO objects of the output images
        """
        thermal_output_filename = ""

        if (minTemp is not None and maxTemp is None) or (
                maxTemp is not None and minTemp is None
        ):
            raise Exception(
                "Specify BOTH a maximum and minimum temperature value, or use the default by specifying neither"
            )
        if maxTemp is not None and minTemp is not None and maxTemp <= minTemp:
            raise Exception("The maxTemp value must be greater than minTemp")

        if self.thermal_image_np is None:
            self.thermal_image_np = self.extract_thermal_image()

        if minTemp is not None and maxTemp is not None:
            thermal_normalized = (self.thermal_image_np - minTemp) / (maxTemp - minTemp)
        else:
            thermal_normalized = (
                                         self.thermal_image_np - np.amin(self.thermal_image_np)
                                 ) / (np.amax(self.thermal_image_np) - np.amin(self.thermal_image_np))

        if not bytesIO:
            thermal_output_filename_array = self.flir_img_filename.split(".")
            thermal_output_filename = (
                    thermal_output_filename_array[0]
                    + "_thermal."
                    + thermal_output_filename_array[1]
            )

        return_array = []
        for palette in self.palettes:
            img_thermal = Image.fromarray(palette(thermal_normalized, bytes=True))
            # convert to jpeg and enhance
            img_thermal = img_thermal.convert("RGB")
            enhancer = ImageEnhance.Sharpness(img_thermal)
            img_thermal = enhancer.enhance(3)

            if bytesIO:
                bytes = io.BytesIO()
                img_thermal.save(bytes, "jpeg", quality=100)
                return_array.append(bytes)
            else:
                transformed_filename = transform_filename_into_directory(thermal_output_filename, str(palette.name))
                filename_array = transformed_filename.split(".")
                filename = (
                        filename_array[0]
                        + "_"
                        + str(palette.name)
                        + "."
                        + filename_array[1]
                )
                if self.is_debug:
                    logger.debug("Saving Thermal image to:{}".format(filename))

                img_thermal.save(filename, "jpeg", quality=100)
                return_array.append(filename)

        return return_array


def transform_filename_into_directory(filename: str, palette: str):
    filename_array = filename.rsplit("/", 1)
    directory = f"{filename_array[0]}/{palette}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/{filename_array[1]}"
    return filename
