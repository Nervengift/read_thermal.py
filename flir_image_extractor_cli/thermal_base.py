"""Thermal base module."""
import io
import json
import subprocess as sp
import sys, os, platform
import requests, zipfile
from pathlib import Path
from threading import Thread

import cv2 as cv
import numpy as np
from logzero import logger
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from . import flyr_unpack
from . import utils as ThermalImageHelpers

plt.style.use("ggplot")


class ThermalImage:
    """Thermal Image class."""

    def __init__(self, image_path, camera_manufacturer, color_map="jet", thermal_np=None):
        """Base Class for Thermal Images.

        Args:
            image_path (str): Path of image to be loaded
            camera_manufacturer (str, optional): Which type of thermal camera was the image captured from.
                Supported values: ["flir","dji"].
            color_map (str, optional): The default colour map to be used when loading the image. Defaults to "jet".
            thermal_np (np.ndarray): Initialize directly with temp array.
        """
        self.image_path = image_path
        # Convert the string false colour map to an opencv object
        self.cmap = getattr(cv, f"COLORMAP_{color_map.upper()}")

        # Load the temperature matrix, sensor matrix and image metadata. First two are loaded as np arrays
        if camera_manufacturer.lower() == "flir":
            self.thermal_np, self.raw_sensor_np, self.meta = self.extract_temperatures_flir()
        elif camera_manufacturer.lower() == "dji":
            self.thermal_np, self.raw_sensor_np, self.meta = self.extract_temperatures_dji()
        elif camera_manufacturer.lower() == "pass":
            pass
        else:
            logger.error(f"Cannot handle data from camera manufacturer {camera_manufacturer}")

        if thermal_np is not None:
            self.thermal_np = thermal_np

        self.global_min_temp = np.min(self.thermal_np)
        self.global_max_temp = np.max(self.thermal_np)

    def extract_temperatures_flir(self):
        """Extracts the FLIR-encoded thermal image as 2D floating-point numpy array with temperatures in degC."""
        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        exif_binary = "exiftool.exe" if "win" in sys.platform else "exiftool"
        meta_json = sp.Popen(
            (
                f'{exif_binary} "{self.image_path}" -Emissivity -ObjectDistance -AtmosphericTemperature '
                "-ReflectedApparentTemperature -IRWindowTemperature -IRWindowTransmission -RelativeHumidity "
                "-PlanckR1 -PlanckB -PlanckF -PlanckO -PlanckR2 -j"
            ),
            shell=True,
            stdout=sp.PIPE,
        ).communicate()[0]

        meta = json.loads(meta_json)[0]

        for key in (
            "ObjectDistance",
            "AtmosphericTemperature",
            "ReflectedApparentTemperature",
            "IRWindowTemperature",
            "RelativeHumidity",
        ):
            meta[key] = ThermalImageHelpers.parse_from_exif_str(meta[key])

        # exifread can't extract the embedded thermal image, use exiftool instead
        # sp popen can't handle bytes

        thermal_img_bytes = sp.check_output([exif_binary, "-RawThermalImage", "-b", f"{self.image_path}"])

        thermal_img_stream = io.BytesIO(thermal_img_bytes)
        thermal_img = Image.open(thermal_img_stream)
        img_format = thermal_img.format

        # checking for the type of the decoded images
        if img_format == "TIFF":
            raw_sensor_np = np.array(thermal_img)
        elif img_format == "PNG":
            raw_sensor_np = flyr_unpack.unpack(str(self.image_path))

        # raw values -> temperature E=meta['Emissivity']
        thermal_np = ThermalImageHelpers.sensor_vals_to_temp(raw_sensor_np, **meta)

        return thermal_np, raw_sensor_np, meta

    def extract_temperatures_dji(self):
        """Extracts the DJI-encoded thermal image as 2D floating-point numpy array with temperatures in degC.

        The raw sensor values are obtained using the sample binaries provided in the official Thermal SDK by DJI.
        The executable file is run and generates a 16 bit unsigned RAW image with Little Endian byte order.
        Link to DJI Forum post: https://forum.dji.com/forum.php?mod=redirect&goto=findpost&ptid=230321&pid=2389016
        """
        # read image metadata for the dji camera images
        exif_binary = "exiftool.exe" if "win" in sys.platform else "exiftool"
        meta_json = sp.Popen(
            (
                f'{exif_binary} "{self.image_path}" -Emissivity -ObjectDistance -AtmosphericTemperature '
                "-ReflectedApparentTemperature -IRWindowTemperature -IRWindowTransmission -RelativeHumidity "
                "-PlanckR1 -PlanckB -PlanckF -PlanckO -PlanckR2 -Make -Model -j"
            ),
            shell=True,
            stdout=sp.PIPE,
        ).communicate()[0]
        meta = json.loads(meta_json)[0]
        camera_model = meta["Model"]

        meta = {
            "Emissivity": 1.0,
            "ObjectDistance": 1,
            "AtmosphericTemperature": 20,
            "ReflectedApparentTemperature": 20,
            "IRWindowTemperature": 20,
            "IRWindowTransmission": 1,
            "RelativeHumidity": 50,
            "PlanckR1": 21106.77,
            "PlanckB": 1501,
            "PlanckF": 1,
            "PlanckO": -7340,
            "PlanckR2": 0.012545258,
        }

        ## calculating the raw sensor values

        # OS and architecture checks
        os_name = "windows" if "win" in sys.platform else "linux"
        architecture_name = "release_x64" if "64bit" in platform.architecture()[0] else "release_x86"
        dji_binary = "dji_irp.exe" if "win" in sys.platform else "dji_irp"
        dji_executables_url = (
            "https://dtpl-ai-public.s3.ap-south-1.amazonaws.com/Thermal_Image_Analysis/DJI_SDK/dji_executables.zip"
        )

        # If DJI executable files aren't present, download them and extract the folder contents.
        if not Path("dji_executables").exists():
            r = requests.get(dji_executables_url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()

        if "linux" in os_name:
            # Linux needs path of libdirp.so file added to Environment variable and Execute permission to executable file.
            path_executable = str(Path("./dji_executables", os_name, architecture_name))
            os.environ["LD_LIBRARY_PATH"] = path_executable
            sp.run(["chmod", "u+x", str(Path(path_executable, dji_binary))])
        elif "windows" in os_name:
            path_executable = str(Path("dji_executables", os_name, architecture_name))

        if "MAVIC" not in camera_model:
            # Run executable file dji_irp passing image path and prevent output printing to console. Raw file generated.
            sp.run(
                [str(Path(path_executable, dji_binary)), "-s", f"{self.image_path}", "-a", "extract"],
                universal_newlines=True,
                stdout=sp.DEVNULL,
                stderr=sp.STDOUT,
            )
            data = Path("output.raw").read_bytes()
            # Read the contents of the generated output.raw file.
            img = Image.frombytes("I;16L", (640, 512), data)
            # After the data is read from the output.raw file, remove the file
            os.remove("output.raw")
        else:
            # Adding support for MAVIC2-ENTERPRISE-ADVANCED Camera images
            im = Image.open(self.image_path)
            # concatenate APP3 chunks of data
            a = im.applist[3][1]
            for i in range(4, 14):
                a += im.applist[i][1]
            # create image from bytes
            img = Image.frombytes("I;16L", (640, 512), a)

        # Extract raw sensor values from generated image into numpy array
        raw_sensor_np = np.array(img)

        ## extracting the temperatures from thermal images
        thermal_np = ThermalImageHelpers.sensor_vals_to_temp(raw_sensor_np, **meta)

        return thermal_np, raw_sensor_np, meta

    def generate_colorbar(self, min_temp=None, max_temp=None, cmap=cv.COLORMAP_JET, height=None):
        """Function to generate a colourbar image that can be stitched to the side of your thermal image.

        Args:
            min_temp (int, optional): Minimum temperature on colourbar.
                If not passed, then the image's minimumtemperature is taken. Defaults to None.
            max_temp (int, optional): Maximum temperature on colourbar.
                If not passed, then the image's maximumtemperature is taken. Defaults to None.
            cmap (cv Colormap, optional): Which false colour mapping to use for the colourbar.
                Defaults to cv.COLORMAP_JET.
            height (int, optional): Height of the colourbar canvas to be used.
                If not passed, takes the height of the thermal image. Defaults to None.

        Returns:
            np.ndarray: A colourbar of the required height with temperature values labelled
        """
        min_temp = self.global_min_temp if min_temp is None else min_temp
        max_temp = self.global_max_temp if max_temp is None else max_temp

        # Build a 255 x 21 colour bar, and create an image to append to the main image
        cb_gray = np.arange(255, 0, -1, dtype=np.uint8).reshape((255, 1))
        cb_gray = np.tile(cb_gray, 21)

        # If the image height happens to be smaller than 255 pixels, then reduce the colour bar accordingly
        # if self.thermal_np.shape[0] < 255:
        #     cb_gray = cv.resize(
        #         cb_gray, (min(21, int(self.thermal_np.shape[1] * 0.05)), int(self.thermal_np.shape[0] * 0.75))
        #     )

        cb_color = cv.applyColorMap(cb_gray, cmap) if cmap is not None else cv.cvtColor(cb_gray, cv.COLOR_GRAY2BGR)

        # Decide layout of the colourbar canvas
        cb_canvas_height_slice = height if height is not None else self.thermal_np.shape[0]
        cb_canvas = np.zeros((cb_canvas_height_slice, cb_color.shape[1] + 30, 3), dtype=np.uint8)
        cb_canvas_start_height = cb_canvas.shape[0] // 2 - cb_color.shape[0] // 2
        cb_height_slice = slice(cb_canvas_start_height, cb_canvas_start_height + cb_color.shape[0])

        # Populate the canvas with the colourbar and relevant text
        cb_canvas[cb_height_slice, 10 : 10 + cb_color.shape[1]] = cb_color
        min_temp_text = (5, cb_canvas_start_height + cb_color.shape[0] + 30)
        max_temp_text = (5, cb_canvas_start_height - 20)
        cv.putText(cb_canvas, str(round(min_temp, 2)), min_temp_text, cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, 8)
        cv.putText(cb_canvas, str(round(max_temp, 2)), max_temp_text, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)

        return cb_canvas

    def save_thermal_image(self, output_path):
        """Converts the floating-point temperature array into an image and saves the image.

        Takes the current `self.thermal_np` temperature array, converts it to image-friendly uint8 format. Then it
        computes and stitches a colourbar for required scale. Finally it saves the image to the specified output path

        Args:
            output_path (str / Path): Save destination for the thermal image
        """
        # Generate a colour mapped image using default colour map
        cmapped_thermal_img = ThermalImageHelpers.get_cmapped_temp_image(self.thermal_np, colormap=self.cmap)
        cb_canvas = self.generate_colorbar(cmap=self.cmap)
        cmapped_thermal_img = cv.hconcat((cmapped_thermal_img, cb_canvas))
        cv.imwrite(str(output_path), cmapped_thermal_img)


class ThermalImageAnnotation:
    """Base Class for Drawing on thermographs."""

    line_flag = False
    contour = []
    drawing = False
    scale_moving = False
    measurement_moving = False
    rect_moving = False
    spots_moving = False
    xo, yo = 0, 0
    xdisp, ydisp = None, None
    measurement_index = None
    rect_index = None
    spots_index = None

    scale_contours = []
    measurement_contours = []
    measurement_rects = []
    emissivity_contours = []
    spots = []

    @staticmethod
    def draw_contour_area(event, x, y, flags, params):
        """Draw contour area."""
        thermal_image = params[0]
        contours = params[1]

        is_rect = params[2][0]
        point1 = params[2][1]
        point2 = params[2][2]

        if event == cv.EVENT_LBUTTONDOWN:
            if not ThermalImageAnnotation.drawing:
                ThermalImageAnnotation.drawing = True
                if is_rect:
                    point1[0] = (x, y)

        elif event == cv.EVENT_MOUSEMOVE:
            if ThermalImageAnnotation.drawing:
                if not is_rect:
                    cv.circle(thermal_image, (x, y), 1, (0, 0, 0), -1)
                    ThermalImageAnnotation.contour.append((x, y))
                else:
                    point2[0] = (x, y)

        elif event == cv.EVENT_LBUTTONUP:
            ThermalImageAnnotation.drawing = False
            ThermalImageAnnotation.contour = np.asarray(ThermalImageAnnotation.contour, dtype=np.int32)
            if len(ThermalImageAnnotation.contour) > 0:
                contours.append(ThermalImageAnnotation.contour)
            ThermalImageAnnotation.contour = []

    @staticmethod
    def draw_spots(event, x, y, flags, params):
        """Draw spots."""
        point = params[0]
        flag = params[1]
        point.clear()

        if event == cv.EVENT_MOUSEMOVE:
            if ThermalImageAnnotation.drawing:
                point.append(x)
                point.append(y)

        elif event == cv.EVENT_LBUTTONDOWN:
            ThermalImageAnnotation.drawing = False
            point.append(x)
            point.append(y)
            flag[0] = False

    @staticmethod
    def get_spots(thermal_image):
        """Get spots."""
        ThermalImageAnnotation.drawing = True
        image_copy = thermal_image.copy()
        original_copy = image_copy.copy()
        if len(original_copy.shape) < 3:
            cmap_copy = cv.applyColorMap(original_copy, cv.COLORMAP_JET)

        point = []
        spot_points = []
        flag = [True]
        cv.namedWindow("Image")
        cv.setMouseCallback("Image", ThermalImageAnnotation.draw_spots, (point, flag))
        while 1:
            image_copy = original_copy.copy()
            for i in range(0, len(spot_points)):
                cv.circle(image_copy, spot_points[i], 5, 0, -1)
                try:
                    cv.circle(cmap_copy, spot_points[i], 5, 0, -1)
                except Exception:
                    cv.circle(original_copy, spot_points[i], 5, 0, -1)

            if len(point) > 0:
                cv.circle(image_copy, tuple(point), 5, 0, -1)

            if not flag[0]:
                spot_points.append(tuple(point))
                flag[0] = True

            cv.imshow("Image", image_copy)
            k = cv.waitKey(1) & 0xFF

            if k in (13, 141):
                break

        ThermalImageAnnotation.drawing = False
        cv.destroyAllWindows()
        # origi_copy = cv.UMat(origi_copy)
        if len(original_copy.shape) == 3:
            gray = cv.cvtColor(original_copy, cv.COLOR_BGR2GRAY)
        else:
            gray = cv.cvtColor(cmap_copy, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ThermalImageAnnotation.spots = contours

    @staticmethod
    def get_spots_values(thermal_np, raw_sensor_np, contours):
        """Get spots values."""
        spots_measurement_values = []
        for i in range(0, len(contours)):
            spots_measurement_values.append(ThermalImageHelpers.get_roi(thermal_np, raw_sensor_np, contours, i)[1])

        return spots_measurement_values

    @staticmethod
    def draw_line(event, x, y, flags, params):
        """Draw line."""
        lp1, lp2 = params[0], params[1]
        thermal_image = params[2]

        if len(lp1) <= 2 and len(lp2) < 2:
            if event == cv.EVENT_LBUTTONDOWN:
                ThermalImageAnnotation.line_flag = not ThermalImageAnnotation.line_flag
                if ThermalImageAnnotation.line_flag:
                    lp1.append(x)
                    lp1.append(y)

                else:
                    lp2.append(x)
                    lp2.append(y)
                    lp1 = tuple(lp1)
                    lp2 = tuple(lp2)
                    cv.line(thermal_image, lp1, lp2, (0, 0, 0), 2, 8)

    @staticmethod
    def get_line(image):
        """Get line."""
        point1 = []
        point2 = []

        cv.namedWindow("image")
        cv.setMouseCallback("image", ThermalImageAnnotation.draw_line, (point1, point2, image))

        while 1:
            cv.imshow("image", image)
            k = cv.waitKey(1) & 0xFF

            if k in (13, 141):
                break

        cv.destroyWindow("image")

        thresh = 15
        line = []
        p1x, p1y = point1[0], point1[1]
        p2x, p2y = point2[0], point2[1]

        if abs((p1x - p2x)) > thresh and abs((p1y - p2y)) > thresh:
            # Using y = mx + c
            m = (p2y - p1y) / (p2x - p1x)
            c = p2y - (m * p2x)
            if p1x > p2x:
                for x in range(p1x, p2x - 1, -1):
                    y = int((m * x) + c)
                    line.append((x, y))
            else:
                for x in range(p1x, p2x + 1):
                    y = int((m * x) + c)
                    line.append((x, y))

        elif abs(p1x - p2x) <= thresh:
            if p1y > p2y:
                for y in range(p1y, p2y - 1, -1):
                    line.append((p1x, y))
            else:
                for y in range(p1y, p2y + 1):
                    line.append((p1x, y))

        else:
            if p1x > p2x:
                for x in range(p1x, p2x - 1, -1):
                    line.append((x, p1y))
            else:
                for x in range(p1x, p2x + 1):
                    line.append((x, p1y))

        return line, (p1x, p1y), (p2x, p2y)

    @staticmethod
    def line_measurement(image, thermal_np, cmap=cv.COLORMAP_JET, plt_title="Linw Plot"):
        """Line measurement."""
        logger.info("Please click on the two extreme points of the line")
        img = image.copy()
        line, point1, point2 = ThermalImageAnnotation.get_line(img)
        line_temps = np.zeros(len(line))

        if len(img.shape) == 3:
            gray_values = np.arange(256, dtype=np.uint8)
            color_values = map(tuple, cv.applyColorMap(gray_values, cmap).reshape(256, 3))
            color_to_gray_map = dict(zip(color_values, gray_values))
            img = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, image)

        for i in range(0, len(line)):
            line_temps[i] = thermal_np[line[i][1], line[i][0]]

        cv.line(img, point1, point2, 255, 2, 8)

        plt.subplot(1, 8, (1, 3))
        plt.imshow(img, cmap="jet")
        plt.title(plt_title)

        a = plt.subplot(1, 8, (5, 8))
        a.set_xlabel("Distance in pixels")
        a.set_ylabel("Temperature in C")
        plt.plot(line_temps)
        if max(line_temps) - min(line_temps) < 5:
            plt.ylim([min(line_temps - 2.5), max(line_temps + 2.5)])
        plt.title("Distance vs Temperature")
        plt.show()

        logger.info(f"\nMin line: {np.amin(line_temps)}\nMax line: {np.amax(line_temps)}")

    @staticmethod
    def is_in_rect(rectangle, point):
        """Check if point in rectangle."""
        tlx, tly, w, h = rectangle
        px, py = point
        is_inside = False
        if tlx < px < tlx + w:
            if tly < py < tly + h:
                is_inside = True
        return is_inside

    @staticmethod
    def move_contours(event, x, y, flags, params):  # scale contour,emissivity contours
        """Move contours."""
        ThermalImageAnnotation.xdisp = None
        ThermalImageAnnotation.ydisp = None
        measurement_contours = params[0]
        measurement_rects = params[1]
        scale_contours = params[2]
        spot_contours = params[3]
        img = params[4]
        vals = params[5]
        spot_vals = params[6]
        scale_contour = []

        if len(scale_contours) > 0:
            scale_contour = scale_contours[0]

        if ThermalImageAnnotation.measurement_moving:
            measurement_cont = measurement_contours[ThermalImageAnnotation.measurement_index]

        if ThermalImageAnnotation.rect_moving:
            measurement_rect = measurement_rects[ThermalImageAnnotation.rect_index]

        if ThermalImageAnnotation.spots_moving:
            spot_cont = spot_contours[ThermalImageAnnotation.spots_index]

        if event == cv.EVENT_RBUTTONDOWN:
            for i in range(0, len(measurement_contours)):
                if cv.pointPolygonTest(measurement_contours[i], (x, y), False) == 1:
                    ThermalImageAnnotation.measurement_index = i
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y
                    ThermalImageAnnotation.measurement_moving = True
                    break

            for i in range(0, len(measurement_rects)):
                if measurement_rects[i][0] <= x <= (measurement_rects[i][0] + measurement_rects[i][2]):
                    if measurement_rects[i][1] <= y <= (measurement_rects[i][1] + measurement_rects[i][3]):
                        ThermalImageAnnotation.rect_index = i
                        ThermalImageAnnotation.xo = x
                        ThermalImageAnnotation.yo = y
                        ThermalImageAnnotation.rect_moving = True
                        break

            if len(scale_contours) > 0:
                if cv.pointPolygonTest(scale_contour, (x, y), False) == 1:
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y
                    ThermalImageAnnotation.scale_moving = True

            for i in range(0, len(spot_contours)):
                if cv.pointPolygonTest(spot_contours[i], (x, y), False) == 1:
                    ThermalImageAnnotation.spots_index = i
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y
                    ThermalImageAnnotation.spots_moving = True
                    break

        elif event == cv.EVENT_MOUSEMOVE:
            if ThermalImageAnnotation.measurement_moving:
                measurement_cont[:, 0] += x - ThermalImageAnnotation.xo
                measurement_cont[:, 1] += y - ThermalImageAnnotation.yo

                if (
                    np.max(measurement_cont[:, 0]) >= img.shape[1]
                    or np.amax(measurement_cont[:, 1]) >= img.shape[0]
                    or np.amin(measurement_cont[:, 0]) <= 0
                    or np.amin(measurement_cont[:, 1]) <= 0
                ):
                    measurement_cont[:, 0] -= x - ThermalImageAnnotation.xo
                    measurement_cont[:, 1] -= y - ThermalImageAnnotation.yo
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y

            if ThermalImageAnnotation.rect_moving is True:
                x_new = measurement_rect[0] + (x - ThermalImageAnnotation.xo)
                y_new = measurement_rect[1] + (y - ThermalImageAnnotation.yo)

                if x_new >= img.shape[1] - measurement_rect[2]:
                    x_new = img.shape[1] - measurement_rect[2] - 1
                if x_new <= 0:
                    x_new = 1
                if y_new >= img.shape[0] - measurement_rect[3]:
                    y_new = img.shape[0] - measurement_rect[3] - 1
                if y_new <= 0:
                    y_new = 1
                measurement_rects[ThermalImageAnnotation.rect_index] = (
                    x_new,
                    y_new,
                    measurement_rect[2],
                    measurement_rect[3],
                )
                ThermalImageAnnotation.xo = x
                ThermalImageAnnotation.yo = y

            if ThermalImageAnnotation.scale_moving:
                scale_contour[:, 0] += x - ThermalImageAnnotation.xo
                scale_contour[:, 1] += y - ThermalImageAnnotation.yo

                if (
                    np.max(scale_contour[:, 0]) >= img.shape[1]
                    or np.amax(scale_contour[:, 1]) >= img.shape[0]
                    or np.amin(scale_contour[:, 0]) <= 0
                    or np.amin(scale_contour[:, 1]) <= 0
                ):
                    scale_contour[:, 0] -= x - ThermalImageAnnotation.xo
                    scale_contour[:, 1] -= y - ThermalImageAnnotation.yo
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y

            if ThermalImageAnnotation.spots_moving:
                spot_cont[:, 0, 0] += x - ThermalImageAnnotation.xo
                spot_cont[:, 0, 1] += y - ThermalImageAnnotation.yo

                if (
                    np.max(spot_cont[:, 0, 0]) >= img.shape[1]
                    or np.amax(spot_cont[:, 0, 1]) >= img.shape[0]
                    or np.amin(spot_cont[:, 0, 0]) <= 0
                    or np.amin(spot_cont[:, 0, 1]) <= 0
                ):
                    spot_cont[:, 0, 0] -= x - ThermalImageAnnotation.xo
                    spot_cont[:, 0, 1] -= y - ThermalImageAnnotation.yo
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    ThermalImageAnnotation.xo = x
                    ThermalImageAnnotation.yo = y

        elif event == cv.EVENT_RBUTTONUP:
            ThermalImageAnnotation.scale_moving = False
            ThermalImageAnnotation.measurement_moving = False
            ThermalImageAnnotation.spots_moving = False
            ThermalImageAnnotation.rect_moving = False

        elif event == cv.EVENT_LBUTTONDBLCLK:
            for i in range(0, len(measurement_contours)):
                if cv.pointPolygonTest(measurement_contours[i], (x, y), False) == 1:
                    logger.info(
                        f"\nMaximum temp: {np.amax(vals[i])}\
                            Minimum temp: {np.amin(vals[i])}\
                            Avg: {np.average(vals[i])}"
                    )
                    maxi = round(np.amax(vals[i]), 2)
                    mini = round(np.amin(vals[i]), 2)
                    avg = round(np.average(vals[i]), 2)

                    with open("config.txt", "w+") as new:
                        new.write("*" + str(maxi) + "*" + "\n")
                        new.write("*" + str(mini) + "*" + "\n")
                        new.write("*" + str(avg) + "*" + "\n")
            for i in range(len(measurement_rects)):
                if ThermalImageAnnotation.is_in_rect(measurement_rects[i], (x, y)):
                    logger.info(
                        f"\nMaximum temp: {np.amax(vals[len(measurement_contours) + i])}\
                            Minimum temp: {np.amin(vals[len(measurement_contours) + i])}\
                            Avg: {np.average(vals[len(measurement_contours) + i])}\n"
                    )  # vals stores free hand values first, and then rects; hence the 'len(measurement_contours) + i'

            for i in range(0, len(spot_contours)):
                if cv.pointPolygonTest(spot_contours[i], (x, y), False) == 1:
                    logger.info(
                        f"\nMaximum temp: {np.amax(spot_vals[i])}\
                            Minimum temp: {np.amin(spot_vals[i])}\
                            Avg: {np.average(spot_vals[i])}\n"
                    )
                    maxi = round(np.amax(spot_vals[i]), 2)
                    mini = round(np.amin(spot_vals[i]), 2)
                    avg = round(np.average(spot_vals[i]), 2)

                    with open("config.txt", "w+") as new:
                        new.write("*" + str(maxi) + "*" + "\n")
                        new.write("*" + str(mini) + "*" + "\n")
                        new.write("*" + str(avg) + "*" + "\n")

        elif event == cv.EVENT_MBUTTONDOWN:
            ThermalImageAnnotation.xdisp = x
            ThermalImageAnnotation.ydisp = y

    @classmethod
    def get_contours(cls, thermal_image, contours, is_rect=False):
        """Get contours."""
        temp_image = thermal_image.copy()
        point1, point2 = [[]], [[]]
        cv.namedWindow("image")
        cv.setMouseCallback("image", cls.draw_contour_area, (temp_image, contours, [is_rect, point1, point2]))

        while 1:
            cv.imshow("image", temp_image)
            if is_rect:
                if len(point1[0]) > 0 and len(point2[0]) > 0:
                    temp_image = cv.rectangle(thermal_image.copy(), point1[0], point2[0], (0, 0, 255))
            k = cv.waitKey(1) & 0xFF

            if k in (13, 141):
                redraw = None
                if is_rect is True and (len(point1[0]) == 0 or len(point2[0]) == 0):
                    logger.warning("No rectangle has been drawn. Do you want to continue?")
                    redraw = input("1-Yes\t0-No,draw rectangle again\n")

                if redraw is not None and redraw == 0:
                    logger.info("Draw a rectangle")
                else:
                    if is_rect is True and redraw is not None:
                        logger.warning("Exiting function without drawing a rectangle")
                        is_rect = False
                    break
        cv.destroyWindow("image")
        if is_rect:
            area_rect = point1[0][0], point1[0][1], abs(point1[0][0] - point2[0][0]), abs(point1[0][1] - point2[0][1])
            return area_rect

        return None

    @staticmethod
    def get_measurement_contours(image, is_rect=False):
        """Get measurement contours."""
        ThermalImageAnnotation.contour = []
        img = image.copy()
        area_rect = ThermalImageAnnotation.get_contours(
            img, ThermalImageAnnotation.measurement_contours, is_rect=is_rect
        )
        if area_rect is not None:
            ThermalImageAnnotation.measurement_rects.append(area_rect)

    @staticmethod
    def get_measurement_areas_values(image, thermal_np, raw_sensor_np, is_rect=False):
        """Get measurement areas values."""
        measurement_areas_thermal_values = []
        measurement_area_indices = []

        for i in range(0, len(ThermalImageAnnotation.measurement_contours)):
            _, thermal_vals, indices = ThermalImageHelpers.get_roi(
                thermal_np, raw_sensor_np, ThermalImageAnnotation.measurement_contours, i
            )
            measurement_areas_thermal_values.append(thermal_vals)
            measurement_area_indices.append(indices)

        # measurement_area_indices = None
        for i in range(0, len(ThermalImageAnnotation.measurement_rects)):
            measurement_areas_thermal_values.append(
                ThermalImageHelpers.get_roi(
                    thermal_np,
                    raw_sensor_np,
                    ThermalImageAnnotation.measurement_contours,
                    i,
                    area_rect=ThermalImageAnnotation.measurement_rects[i],
                )[1]
            )
        return measurement_areas_thermal_values, measurement_area_indices

    @staticmethod
    def get_scaled_image(img, thermal_np, raw_sensor_np, cmap=cv.COLORMAP_JET, is_rect=False):
        """Get scaled image."""
        ThermalImageAnnotation.scale_contours = []
        ThermalImageAnnotation.contour = []
        ThermalImageAnnotation.get_contours(img, ThermalImageAnnotation.scale_contours)
        flag = False

        if len(ThermalImageAnnotation.scale_contours) > 0:

            if len(ThermalImageAnnotation.scale_contours[0]) > 15:
                flag = True
                thermal_roi_values = ThermalImageHelpers.get_roi(
                    thermal_np, raw_sensor_np, ThermalImageAnnotation.scale_contours, 0
                )[1]
                temp_scaled = ThermalImageHelpers.scale_with_roi(thermal_np, thermal_roi_values)
                temp_scaled_image = ThermalImageHelpers.get_cmapped_temp_image(temp_scaled, colormap=cmap)

        if not flag:
            temp_scaled = thermal_np.copy()
            temp_scaled_image = ThermalImageHelpers.get_cmapped_temp_image(temp_scaled, colormap=cmap)

        return temp_scaled, temp_scaled_image

    @staticmethod
    def get_emissivity_changed_image(img, thermal_np, raw_sensor_np, meta, cmap=cv.COLORMAP_JET):
        """Change emmisivity value for a marked region in image."""
        ThermalImageAnnotation.contour = []
        ThermalImageAnnotation.emissivity_contours = []
        ThermalImageAnnotation.get_contours(img, ThermalImageAnnotation.emissivity_contours)
        flag = False

        if len(ThermalImageAnnotation.emissivity_contours) > 0:
            if len(ThermalImageAnnotation.emissivity_contours[0]) >= 15:
                flag = True
                raw_roi_values, _, indices = ThermalImageHelpers.get_roi(
                    thermal_np, raw_sensor_np, ThermalImageAnnotation.emissivity_contours, 0
                )
                emissivity_changed_array = ThermalImageHelpers.change_emissivity_for_roi(
                    thermal_np, meta, ThermalImageAnnotation.emissivity_contours, raw_roi_values, indices
                )
                emissivity_changed_image = ThermalImageHelpers.get_cmapped_temp_image(
                    emissivity_changed_array, colormap=cmap
                )

        if not flag:
            emissivity_changed_array = thermal_np
            emissivity_changed_image = img

        ThermalImageAnnotation.emissivity_contours = []
        return emissivity_changed_array, emissivity_changed_image


class ThermalSeqVideo:
    r"""Base class for splitting SEQ files into multiple fff and jpg files.

    refer: https://exiftool.org/forum/index.php?topic=5279.0
    @purpose:
      Read .seq files from Flir IR camera and write each frame to temporary binary file.

    @usage:
      seqToBin.py _FILE_NAME_.seq

    @note:
      When first using this code for a new camera, it might need find the bits separating
      each frame, which is possibly IR camera specific. Please run:
        hexdump -n16 -C _FILE_NAME_.seq

      @@Example
        >$ hexdump -n16 -C Rec-000667_test.seq
        00000000  46 46 46 00 52 65 73 65  61 72 63 68 49 52 00 00  |FFF.ResearchIR..|
        00000010
        So, for this camera, the separation patten is:
        \x46\x46\x46\x00\x52\x65\x73\x65\x61\x72\x63\x68\x49\x52
        which == FFFResearchIR

    P.S. Runs much faster when writing data to an empty folder rather than rewriting existing folder's files
    """

    def __init__(self, input_video):
        """Initializer for ThermalSeqVideo."""
        self.split_thermal(input_video)

    @staticmethod
    def __get_hex_sep_pattern(input_video):
        r"""Function to get the hex separation pattern from the seq file automatically.

        The split, and replace functions might have to be modified.
        This hasn't been tried with files other than from the Zenmuse XT2
        Information on '\\x':
        https://stackoverflow.com/questions/2672326/what-does-a-leading-x-mean-in-a-python-string-xaa
        https://www.experts-exchange.com/questions/26938912/Get-rid-of-escape-character.html
        Python eval() function:
        https://www.geeksforgeeks.org/eval-in-python
        """
        pat = sp.check_output(["hexdump", "-n16", "-C", str(input_video)])
        pat = pat.decode("ascii")
        # Following lines are to get the marker (pattern) to the appropriate hex form
        pat = pat.split("00000000 ")[1]
        pat = pat.split("  |")[0]
        pat = pat.replace("  ", " ")
        pat = pat.replace(" ", "\\x")
        pat = f"'{pat}'"
        pat = eval(pat)  # eval is apparently risky to use. Change later
        return pat

    @staticmethod
    def __split_by_marker(f, marker="", block_size=10240):
        current = ""
        bolStartPos = True
        while True:
            block = f.read(block_size)
            if not block:  # end-of-file
                yield marker + current
                return
            block = block.decode("latin-1")
            current += block
            while True:
                markerpos = current.find(marker)
                if bolStartPos:
                    current = current[markerpos + len(marker) :]
                    bolStartPos = False
                elif markerpos < 0:
                    break
                else:
                    yield marker + current[:markerpos]
                    current = current[markerpos + len(marker) :]

    def split_thermal(self, input_video, output_folder=None, path_to_base_thermal_class_folder="."):
        """Splits the thermal SEQ file into separate 'fff' frames by its hex separator pattern.

        (TO DO: Find out more about how exactly this is done)
        Inputs: 'input_video':thermal SEQ video, 'output_folder': Path to output folder
            (Creates folder if it doesn't exist)
        The Threading makes all the cores run at 100%, but it gives ~x4 speed-up.
        """
        if output_folder is None:
            output_folder = Path(input_video).with_suffix("")

        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        sys.path.insert(0, path_to_base_thermal_class_folder)

        idx = 0
        inputname = input_video
        pat = self.__get_hex_sep_pattern(input_video)
        for line in tqdm(self.__split_by_marker(open(inputname, "rb"), marker=pat)):
            outname = output_folder / f"frame_{idx}.fff"
            with open(outname, "wb") as output_file:
                line = line.encode("latin-1")
                output_file.write(line)
            Thread(
                target=get_thermal_image_from_file, kwargs={"thermal_class": ThermalImage, "thermal_input": outname}
            ).start()
            idx = idx + 1
            if idx % 100000 == 0:
                print(f"running index : {idx} ")
                break
        return True

    @staticmethod
    def split_visual(visual_video, fps, fps_ratio, output_folder="visual_frames"):
        """Splits video into frames based on the actual fps, and time between frames of the thermal sequence.

        There is a sync issue where the thermal fps, and visual fps don't have an integer LCM/if LCM is v large.
        Have to try motion interpolation to fix this.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        vid = cv.VideoCapture(visual_video)
        total_frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
        current_frame = 0
        thermal_fps = fps * (1 / fps_ratio)
        thermal_time = 1 / thermal_fps
        logger.info(f"Time between frames for Thermal SEQ: {thermal_time}")
        # Uncomment below lines if you need total time of visual video
        # vid.set(cv.CAP_PROP_POS_AVI_RATIO,1)
        # total_time = vid.get(cv.CAP_PROP_POS_MSEC)
        last_save_time = -1 * thermal_time  # So that it saves the 0th frame
        idx = 0
        while current_frame < total_frames:
            current_frame = vid.get(cv.CAP_PROP_POS_FRAMES)
            try:
                current_time = (1 / fps) * current_frame
            except Exception:
                current_time = 0
            ret, frame = vid.read()
            if ret:
                if (current_time - last_save_time) * 1000 >= ((thermal_time * 1000) - 5):
                    # logger.info(f'Current Time: {current_time}  Last save time: {last_save_time}')
                    cv.imwrite(str(output_folder / f"{idx}.jpg"), frame)
                    idx += 1
                    last_save_time = current_time
        return True


def get_thermal_image_from_file(thermal_input, thermal_class=ThermalImage, colormap=None):
    """
    Function to get the image associated with each RJPG file using the FLIR Thermal base class ThermalImage.

    Saves the thermal images in the same place as the original RJPG
    """
    CThermal = thermal_class

    inputpath = Path(thermal_input)
    if Path.is_dir(inputpath):
        rjpg_img_paths = list(Path(inputpath).glob("*R.JPG"))
        fff_file_paths = list(Path(inputpath).glob("*.fff"))
        if len(rjpg_img_paths) > 0:
            for rjpg_img in tqdm(rjpg_img_paths, total=len(rjpg_img_paths)):
                thermal_obj = CThermal(rjpg_img, color_map=colormap)
                path_wo_ext = str(rjpg_img).replace("_R" + rjpg_img.suffix, "")
                thermal_obj.save_thermal_image(path_wo_ext + ".jpg")

        elif len(fff_file_paths) > 0:
            for fff in tqdm(fff_file_paths, total=len(fff_file_paths)):
                save_image_path = str(fff).replace(".fff", ".jpg")
                thermal_obj = CThermal(fff, color_map=colormap)
                thermal_obj.save_thermal_image(save_image_path)
        else:
            logger.error("Input folder contains neither fff or RJPG files")

    elif Path.is_file(inputpath):
        thermal_obj = CThermal(thermal_input, color_map=colormap)
        path_wo_ext = Path.as_posix(inputpath).replace(inputpath.suffix, "")
        thermal_obj.save_thermal_image(path_wo_ext + ".jpg")

    else:
        logger.error("Path given is neither file nor folder. Please check")


if __name__ == "__main__":
    img_name = Path(sys.argv[1])
    cam_manfacturer = sys.argv[2]
    t_img = ThermalImage(img_name, cam_manfacturer)
    t_img.save_thermal_image(img_name.with_name(f"{img_name.stem}_jet_cmap{img_name.suffix}"))
