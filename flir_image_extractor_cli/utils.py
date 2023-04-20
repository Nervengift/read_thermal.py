"""Thermal Image manipulation utilities."""
import cv2 as cv
import numpy as np
from matplotlib import cm


def sensor_vals_to_temp(
    raw,
    Emissivity=1.0,
    ObjectDistance=1,
    AtmosphericTemperature=20,
    ReflectedApparentTemperature=20,
    IRWindowTemperature=20,
    IRWindowTransmission=1,
    RelativeHumidity=50,
    PlanckR1=21106.77,
    PlanckB=1501,
    PlanckF=1,
    PlanckO=-7340,
    PlanckR2=0.012545258,
    **kwargs,
):
    """Convert raw values from the thermographic sensor sensor to temperatures in °C. Tested for Flir cams."""
    # this calculation has been ported to python from https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
    # a detailed explanation of what is going on here can be found there

    # constants
    ATA1 = 0.006569
    ATA2 = 0.01262
    ATB1 = -0.002276
    ATB2 = -0.00667
    ATX = 1.9

    # transmission through window (calibrated)
    emiss_wind = 1 - IRWindowTransmission
    refl_wind = 0
    # transmission through the air
    h2o = (RelativeHumidity / 100) * np.exp(
        1.5587
        + 0.06939 * (AtmosphericTemperature)
        - 0.00027816 * (AtmosphericTemperature) ** 2
        + 0.00000068455 * (AtmosphericTemperature) ** 3
    )
    tau1 = ATX * np.exp(-np.sqrt(ObjectDistance / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
        -np.sqrt(ObjectDistance / 2) * (ATA2 + ATB2 * np.sqrt(h2o))
    )
    tau2 = ATX * np.exp(-np.sqrt(ObjectDistance / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
        -np.sqrt(ObjectDistance / 2) * (ATA2 + ATB2 * np.sqrt(h2o))
    )
    # radiance from the environment
    raw_refl1 = PlanckR1 / (PlanckR2 * (np.exp(PlanckB / (ReflectedApparentTemperature + 273.15)) - PlanckF)) - PlanckO
    raw_refl1_attn = (1 - Emissivity) / Emissivity * raw_refl1  # Reflected component

    raw_atm1 = (
        PlanckR1 / (PlanckR2 * (np.exp(PlanckB / (AtmosphericTemperature + 273.15)) - PlanckF)) - PlanckO
    )  # Emission from atmosphere 1
    raw_atm1_attn = (1 - tau1) / Emissivity / tau1 * raw_atm1  # attenuation for atmospheric 1 emission

    raw_wind = (
        PlanckR1 / (PlanckR2 * (np.exp(PlanckB / (IRWindowTemperature + 273.15)) - PlanckF)) - PlanckO
    )  # Emission from window due to its own temp
    raw_wind_attn = (
        emiss_wind / Emissivity / tau1 / IRWindowTransmission * raw_wind
    )  # Componen due to window emissivity

    raw_refl2 = (
        PlanckR1 / (PlanckR2 * (np.exp(PlanckB / (ReflectedApparentTemperature + 273.15)) - PlanckF)) - PlanckO
    )  # Reflection from window due to external objects
    raw_refl2_attn = (
        refl_wind / Emissivity / tau1 / IRWindowTransmission * raw_refl2
    )  # component due to window reflectivity

    raw_atm2 = (
        PlanckR1 / (PlanckR2 * (np.exp(PlanckB / (AtmosphericTemperature + 273.15)) - PlanckF)) - PlanckO
    )  # Emission from atmosphere 2
    raw_atm2_attn = (
        (1 - tau2) / Emissivity / tau1 / IRWindowTransmission / tau2 * raw_atm2
    )  # attenuation for atmospheric 2 emission

    raw_obj = (
        raw / Emissivity / tau1 / IRWindowTransmission / tau2
        - raw_atm1_attn
        - raw_atm2_attn
        - raw_wind_attn
        - raw_refl1_attn
        - raw_refl2_attn
    )
    val_to_log = PlanckR1 / (PlanckR2 * (raw_obj + PlanckO)) + PlanckF
    if any(val_to_log.ravel() < 0):
        raise Exception("Image seems to be corrupted")
    # temperature from radiance
    return PlanckB / np.log(val_to_log) - 273.15


def parse_from_exif_str(temp_str):
    """String to float parser."""
    # we assume degrees celsius for temperature, metres for length
    if isinstance(temp_str, str):
        return float(temp_str.split()[0])
    return float(temp_str)


def normalize_temp_matrix(thermal_np):
    """Normalize a temperature matrix to the 0-255 uint8 image range."""
    num = thermal_np - np.amin(thermal_np)
    den = np.amax(thermal_np) - np.amin(thermal_np)
    thermal_np = num / den
    return thermal_np


def get_cmapped_temp_image(thermal_np, colormap=cv.COLORMAP_JET):
    """Converts a temperature matrix into a numpy image."""
    thermal_np_norm = normalize_temp_matrix(thermal_np)
    thermal_image = np.array(thermal_np_norm * 255, dtype=np.uint8)
    if colormap is not None:
        thermal_image = cv.applyColorMap(thermal_image, colormap)
    return thermal_image


def cmap_matplotlib(frame, cmap="jet"):
    """Returns color-mapped image for a given float array."""
    frame_x = frame.copy()
    frame_x -= np.min(frame_x)
    frame_x /= np.max(frame_x)

    return (255 * getattr(cm, cmap)(frame_x)[..., 2::-1]).astype(np.uint8)


def get_temp_image(thermal_np, colormap=cv.COLORMAP_JET):
    """Alias for get_cmapped_temp_image, to be deprecated in the future."""
    return get_cmapped_temp_image(thermal_np, colormap)


def get_roi(thermal_np, raw_sensor_np, Contours, index, area_rect=None):
    """Get values within roi.

    Fetches the sensor values, temperature values and indices (indices within bounding rect)
    of an RoI within a temperature + sensor array.
    """
    raw_roi_values = []
    thermal_roi_values = []
    indices = []

    if area_rect is None:
        img2 = np.zeros((thermal_np.shape[0], thermal_np.shape[1], 1), np.uint8)
        cv.drawContours(img2, Contours, index, 255, -1)
        x, y, w, h = cv.boundingRect(Contours[index])

        indices = np.arange(w * h)
        ind = np.where(img2[:, :, 0] == 255)
        indices = indices[np.where(img2[y : y + h, x : x + w, 0].flatten() == 255)]
        raw_roi_values = raw_sensor_np[ind]
        thermal_roi_values = thermal_np[ind]

    else:
        x, y, w, h = area_rect
        raw_roi_values = raw_sensor_np[y : y + h, x : x + w]
        thermal_roi_values = thermal_np[y : y + h, x : x + w]

    return raw_roi_values, thermal_roi_values, indices


def clip_temp_to_roi(thermal_np, thermal_roi_values):
    """
    Given an RoI within a temperature matrix, this function clips the temperature values in the entire thermal.

    Image temperature values above and below the max/min temperatures within the RoI are clipped to said max/min.

    Args:
        thermal_np (np.ndarray): Floating point array containing the temperature matrix.
        thermal_roi_values (np.ndarray / list): Any iterable containing the temperature values within the RoI.

    Returns:
        np.ndarray: The clipped temperature matrix.
    """
    maximum = np.amax(thermal_roi_values)
    minimum = np.amin(thermal_roi_values)
    thermal_np[thermal_np > maximum] = maximum
    thermal_np[thermal_np < minimum] = minimum
    return thermal_np


def scale_with_roi(thermal_np, thermal_roi_values):
    """Alias for clip_temp_to_roi, to be deprecated in the future."""
    return clip_temp_to_roi(thermal_np, thermal_roi_values)


def coordinates_in_poly(polygon_points, frame_shape=(512, 640)):
    """Returns indices of points within a polygon specified by polygon_points.

    Args:
        polygon_points (list): list of ordered coordinates.

    Returns:
        tuple: Indices of points within a polygon.
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv.fillPoly(mask, pts=[np.array(polygon_points)], color=255)
    return np.where(mask > 0)


def change_emissivity_for_roi(
    thermal_np,
    meta,
    roi_contours,
    raw_roi_values,
    indices,
    new_emissivity=None,
    ref_temperature=None,
    atm_temperature=None,
    np_indices=False,
):
    """Changes the Emissivity for pixels in a given region (RoI) and returns the new temperature matrix."""
    if np_indices:
        meta_x = meta.copy()
        if new_emissivity not in (None, ""):
            meta_x["Emissivity"] = float(new_emissivity)
        if ref_temperature not in (None, ""):
            meta_x["ReflectedApparentTemperature"] = float(ref_temperature)
        if atm_temperature not in (None, ""):
            meta_x["AtmosphericTemperature"] = float(atm_temperature)

        return sensor_vals_to_temp(raw_roi_values, **meta_x)

    x, y, w, h = cv.boundingRect(roi_contours[0])
    temp_array = thermal_np.copy()
    roi_rectangle = temp_array[y : y + h, x : x + w]
    roi_rectangle_flat = roi_rectangle.flatten()
    roi_raw_sensor_np = np.asarray(raw_roi_values)

    if new_emissivity is None:
        new_emissivity = float(input("Enter new Emissivity: "))

    meta_x = meta.copy()
    meta_x["Emissivity"] = new_emissivity

    raw_roi_values_thermal_np = sensor_vals_to_temp(roi_raw_sensor_np, **meta_x)

    count = 0
    for i in range(0, len(roi_rectangle_flat)):
        if i in indices:
            roi_rectangle_flat[i] = raw_roi_values_thermal_np[count]
            count += 1

    temp_array[y : y + h, x : x + w] = roi_rectangle_flat.reshape((h, w))
    return temp_array


def default_scaling_image(array, cmap=cv.COLORMAP_JET):
    """Scales temperature vals for the whole image to be clipped to ~min and max around the mid line."""
    thermal_np = array.copy()
    mid_thermal_np = thermal_np[10 : thermal_np.shape[0] - 10, (int)(thermal_np.shape[1] / 2)]
    maximum = np.amax(mid_thermal_np)
    minimum = np.amin(mid_thermal_np)

    thermal_np[thermal_np > maximum + 10] = maximum + 10
    thermal_np[thermal_np < minimum - 5] = minimum - 5
    image = get_cmapped_temp_image(thermal_np, colormap=cmap)

    return image, thermal_np
