from prompt_toolkit.shortcuts import checkboxlist_dialog, button_dialog, input_dialog, yes_no_dialog, ProgressBar


def path_dialog() -> input_dialog:
    """
    Displays a dialog for the user to input the directory containing the thermal images
    :return:
    @author Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    return input_dialog(
        title="File or Directory Path", text="Input file or directory path: ",
    ).run()


def output_options_checklist() -> checkboxlist_dialog:
    """
    Displays a dialog for the user to input the output options they want
    :return:
    @author Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    return checkboxlist_dialog(
        title="CheckboxList dialog",
        text="Select one or more output options (press Tab to select Ok/Cancel)",
        values=[
            # ("csv", "CSV file containing temperature data (in degrees celsius)"),
            ("bwr", "Thermal image using bwr colormap"),
            ("gnuplot2", "Thermal image gnuplot2 colormap"),
            ("gist_ncar", "Thermal image gist_ncar colormap"),
        ],
    ).run()


def metadata_dialog() -> yes_no_dialog:
    """
    Displays a dialog for the user to specify whether they would like metadata from the original images copied to the
    new ones
    :return:
    @author Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    return yes_no_dialog(
        title="Metadata",
        text="Do you want all of the metadata from the original images copied to the new ones?",
    ).run()


def custom_temperature_dialog() -> button_dialog:
    return button_dialog(
        title="Min and Max Temperature values",
        text="Do you want to set custom min and max temperature values or infer them from the images? You must enter "
             "both a min and max.",
        buttons=[
            ('Infer', False),
            ('Custom', True),
        ]
    ).run()


def custom_temperature_input(value: str) -> input_dialog:
    return input_dialog(
            title=f"Custom {value}",
            text=f"Input custom {value} temperature value (in degrees celsius): ",
        ).run()
