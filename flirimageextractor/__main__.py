from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.shortcuts import input_dialog
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import time

# style that all of the dialogs use
dialog_style = Style.from_dict(
    {
        "dialog": "bg:#88ff88",
        "dialog frame.label": "bg:#ffffff #000000",
        "dialog.body": "bg:#000000 #00ff00",
        "dialog shadow": "bg:#00aa00",
    }
)


path = input_dialog(
    title="File or Directory Path",
    text="Input file or directory path: ",
    style=dialog_style,
).run()

results_array = checkboxlist_dialog(
    title="CheckboxList dialog",
    text="Select one or more output options (press tab to select ok/cancel)",
    values=[
        ("csv", "CSV file containing temperature data (in degrees celcius)"),
        ("bwr", "Thermal image using bwr colormap"),
        ("gnuplot2", "Thermal image gnuplot2 colormap"),
        ("gist_ncar", "Thermal image gist_ncar colormap"),
        ("custom", "Thermal image using a custom colormap"),
    ],
    style=dialog_style,
).run()

if "custom" in results_array:
    custom_colormarp = input_dialog(
        title="Custom Matplotlib Colormap",
        text="Matplotlib colormap name: ",
        style=dialog_style,
    ).run()


# todo: only ask if the user has selected a colormap
metadata = yes_no_dialog(
    title="Metadata",
    text="Do you want all of the metadata from the original images copied to the new ones?",
    style=dialog_style,
).run()

custom_min_max = yes_no_dialog(
    title="Custom Min and Max",
    text="Do you want to set custom min and max values?",
    style=dialog_style,
).run()

if custom_min_max:
    custom_min = input_dialog(
        title="Custom Min",
        text="Input custom minimum temperature value (in degrees celcius): ",
        style=dialog_style,
    ).run()

    custom_max = input_dialog(
        title="Custom Max",
        text="Input custom maximum temperature value (in degrees celcius): ",
        style=dialog_style,
    ).run()

title = HTML("Processing the images...")
with ProgressBar(title=title) as pb:
    for i in pb(range(800)):
        time.sleep(0.01)

