import subprocess
from os import listdir, mkdir
from os.path import isfile, join

FOLDER_PATH = "/Volumes/Jaimyn USB/veolia/thermal/"

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
