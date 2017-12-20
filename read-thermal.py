#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
import subprocess
import io
import json
from math import sqrt,exp,log

def raw2temp(raw, E=1,OD=1,RTemp=20,ATemp=20,IRWTemp=20,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258):
    """ convert raw values from the flir sensor to temperatures in °C """
    # this calculation has been ported to python from https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
    # a detailed explanation of what is going on here can be found there

    # constants
    ATA1=0.006569; ATA2=0.01262; ATB1=-0.002276; ATB2=-0.00667; ATX=1.9

    # transmission through window (calibrated)
    emiss_wind = 1 - IRT
    refl_wind = 0

    # transmission through the air
    h2o = (RH/100)*exp(1.5587+0.06939*(ATemp)-0.00027816*(ATemp)**2+0.00000068455*(ATemp)**3)
    tau1 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
    tau2 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))

    # radiance from the environment
    raw_refl1 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO
    raw_refl1_attn = (1-E)/E*raw_refl1
    raw_atm1 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO
    raw_atm1_attn = (1-tau1)/E/tau1*raw_atm1
    raw_wind = PR1/(PR2*(exp(PB/(IRWTemp+273.15))-PF))-PO
    raw_wind_attn = emiss_wind/E/tau1/IRT*raw_wind
    raw_refl2 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO
    raw_refl2_attn = refl_wind/E/tau1/IRT*raw_refl2
    raw_atm2 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO
    raw_atm2_attn = (1-tau2)/E/tau1/IRT/tau2*raw_atm2
    raw_obj = (raw/E/tau1/IRT/tau2-raw_atm1_attn-raw_atm2_attn-raw_wind_attn-raw_refl1_attn-raw_refl2_attn)

    # temperature from radiance
    temp_C = PB/log(PR1/(PR2*(raw_obj+PO))+PF)-273.15

    return temp_C

def _parse_temp(temp_str):
    # TODO: do this right
    # we assume degrees celsius
    return (float(temp_str.split()[0]))

def _parse_length(length_str):
    # TODO: do this right
    # we assume meters
    return (float(length_str.split()[0]))

def _parse_percent(percentage_str):
    return (float(percentage_str.split()[0]))

def extract_thermal(flir_image_path):
    """ extracts the thermal image as 2D numpy array with temperatures in °C """

    # read image metadata needed for conversion of the raw sensor values
    # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
    meta_json = subprocess.check_output(['exiftool', flir_image_path, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature', '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity', '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
    meta = json.loads(meta_json)[0]

    #exifread can't extract the embedded thermal image, use exiftool instead
    thermal_img_bytes = subprocess.check_output(["exiftool", "-RawThermalImage", "-b", flir_image_path])
    thermal_img_stream = io.BytesIO(thermal_img_bytes)

    thermal_img = Image.open(thermal_img_stream)
    thermal_np = np.array(thermal_img)

    # fix endianness, the bytes in the embedded png are in the wrong order
    thermal_np = np.vectorize(lambda x: (x>>8) + ((x&0x00ff) << 8))(thermal_np)

    # raw values -> temperature
    raw2tempfunc = np.vectorize(lambda x: raw2temp(x, E=meta['Emissivity'], OD=_parse_length(meta['SubjectDistance']), RTemp=_parse_temp(meta['ReflectedApparentTemperature']), ATemp=_parse_temp(meta['AtmosphericTemperature']), IRWTemp=_parse_temp(meta['IRWindowTemperature']), IRT=meta['IRWindowTransmission'], RH=_parse_percent(meta['RelativeHumidity']), PR1=meta['PlanckR1'], PB=meta['PlanckB'], PF=meta['PlanckF'], PO=meta['PlanckO'], PR2=meta['PlanckR2']))
    thermal_np = raw2tempfunc(thermal_np)

    return thermal_np

def extract_visual(flir_image_path):
    """ extracts the visual image as 2D numpy array of RGB values """

    visual_img_bytes = subprocess.check_output(["exiftool", "-EmbeddedImage", "-b", flir_image_path])
    visual_img_stream = io.BytesIO(visual_img_bytes)

    visual_img = Image.open(visual_img_stream)
    visual_np = np.array(visual_img)

    return visual_np


if __name__ == "__main__":

    imgpath = sys.argv[1]

    thermal_np = extract_thermal(imgpath)
    print("max: {}".format(np.amax(thermal_np)))
    print("min: {}".format(np.amin(thermal_np)))

    visual_np = extract_visual(imgpath)

    plt.subplot(1, 2, 1)
    plt.imshow(thermal_np, cmap='hot')
    plt.subplot(1, 2, 2)
    plt.imshow(visual_np)
    plt.show()

    img_visual = Image.fromarray(visual_np)
    thermal_normalized = (thermal_np-np.amin(thermal_np))/(np.amax(thermal_np)-np.amin(thermal_np))
    img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized)*255))

    fn_prefix, _ = os.path.splitext(imgpath)
    img_visual.save(fn_prefix + "_visual.jpg")
    img_thermal.save(fn_prefix + "_thermal.png")
