
def parse_input_args(command_line_args):

    if len(command_line_args) > 1:
        raise ValueError("Too many input arguments provided")

    if len(command_line_args) == 0:
        return 0

    task = command_line_args[0]

    if task == 'mth_usage':
        return 1
    elif task == 'mth_def':
        return 2
    else:
        raise ValueError("Invalid argument entered. Expecting: \'mth_usage\' or \'mth_def\'...")

