import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image


def _extract_float(float_string: str) -> float:
    """Extract the float value of a string, helpful for parsing the exiftool data."""
    digits = re.findall(r"[-+]?\d*\.\d+|\d+", float_string)
    return float(digits[0])


def raw2temp(
    raw: npt.NDArray[np.float64],
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
) -> npt.NDArray[np.float64]:
    """
    Convert raw values from the flir sensor to temperatures in celsius.

    This calculation has been ported to python from
    https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
    a detailed explanation of what is going on here can be found there.
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
    h2o = (RH / 100) * math.exp(
        1.5587
        + 0.06939 * (ATemp)
        - 0.00027816 * (ATemp) ** 2
        + 0.00000068455 * (ATemp) ** 3
    )
    tau1 = ATX * math.exp(-math.sqrt(OD / 2) * (ATA1 + ATB1 * math.sqrt(h2o))) + (
        1 - ATX
    ) * math.exp(-math.sqrt(OD / 2) * (ATA2 + ATB2 * math.sqrt(h2o)))
    tau2 = ATX * math.exp(-math.sqrt(OD / 2) * (ATA1 + ATB1 * math.sqrt(h2o))) + (
        1 - ATX
    ) * math.exp(-math.sqrt(OD / 2) * (ATA2 + ATB2 * math.sqrt(h2o)))

    # radiance from the environment
    raw_refl1 = PR1 / (PR2 * (math.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl1_attn = (1 - E) / E * raw_refl1
    raw_atm1 = PR1 / (PR2 * (math.exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
    raw_wind = PR1 / (PR2 * (math.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
    raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
    raw_refl2 = PR1 / (PR2 * (math.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
    raw_atm2 = PR1 / (PR2 * (math.exp(PB / (ATemp + 273.15)) - PF)) - PO
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


class ThermalExtractor:
    """Extract temperatures for each pixel."""

    def __init__(self, exiftool_path: str = "exiftool") -> None:
        self.exiftool_path = exiftool_path

    def process_image(self, flir_image_path: str) -> npt.NDArray[np.float64]:
        """
        Extract real thermal values from an image file.

        Parameters
        ----------
        flir_image_path
            Path to a FLIR image file

        Returns
        -------
        thermal
            A NumPy array where each pixel value is a temperature in celsius
        """
        if not Path(flir_image_path).exists():
            raise ValueError("Input file does not exist.")

        if self.get_image_type(flir_image_path).upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            fix_endian = False
        else:
            fix_endian = True

        return self.extract_thermal_image(flir_image_path, fix_endian=fix_endian)

    def get_image_type(self, flir_image_path: str) -> str:
        """Get the embedded thermal image type (generally  TIFF or PNG)."""
        meta_json = subprocess.check_output(
            [self.exiftool_path, "-RawThermalImageType", "-j", flir_image_path]
        )
        meta = json.loads(meta_json.decode())[0]
        return meta["RawThermalImageType"]

    def extract_thermal_image(
        self,
        flir_image_path: str,
        default_distance: float = 1.0,
        fix_endian: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Extract the thermal image as 2D numpy array with temperatures in celsius."""
        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [
                self.exiftool_path,
                flir_image_path,
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
        meta = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output(
            [self.exiftool_path, "-RawThermalImage", "-b", flir_image_path]
        )
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = default_distance
        if "SubjectDistance" in meta:
            subject_distance = _extract_float(meta["SubjectDistance"])

        if fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00FF) << 8))(
                thermal_np
            )

        return raw2temp(
            thermal_np,
            E=meta["Emissivity"],
            OD=subject_distance,
            RTemp=_extract_float(meta["ReflectedApparentTemperature"]),
            ATemp=_extract_float(meta["AtmosphericTemperature"]),
            IRWTemp=_extract_float(meta["IRWindowTemperature"]),
            IRT=meta["IRWindowTransmission"],
            RH=_extract_float(meta["RelativeHumidity"]),
            PR1=meta["PlanckR1"],
            PB=meta["PlanckB"],
            PF=meta["PlanckF"],
            PO=meta["PlanckO"],
            PR2=meta["PlanckR2"],
        )
