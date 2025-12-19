from pathlib import Path
import json


# %% operations
def save_options(input_file, suffix, operation=None, overwrite=False):
    """
    input_file: target file
    suffix:
    operation: default '.copy'
    overwrite: default False
    """
    input_file = Path(input_file).with_suffix(suffix)
    if operation is not None:
        input_file = add_extension(input_file, operation)
        # session.operation.suffix
    if overwrite:
        output_file = input_file
        # session.suffix
        # or session.operation.suffix if operation was specified
    else:
        output_file = not_overwrite(input_file)
        # add (count) to avoide overwriting
    return output_file


def add_extension(file, ext: str, suffix=None):
    if suffix is None:
        suffix = Path(file).suffix
    # assert (suffix[0] == '.') and (ext[0] == '.'), \
    # print('invalid suffix; add a dot "." before the extension')
    return Path(file).with_suffix(ext + suffix)


def not_overwrite(file):
    """
    file: pathlib.Path
      path to file which will be avoided to overwrite by adding .copy{i}.`suffix`
      where i starts from None and then will increase from 1 until the target filename does not already exist.

    suffix: str
      extension of the target file, remember to add a dot before the extension, e.g. .dat, .xml
    """
    file = Path(file)
    icopy = 1
    lenstr_icopy = None
    while file.is_file():
        file = file.parent / (f"{file.stem[:lenstr_icopy]}({str(icopy)}){file.suffix}")
        lenstr_icopy = -(len(str(icopy)) + 2)  # +2 for parantheses
        icopy += 1

    return file


# %% meta/log file
def save_log(log_dict, log_file, extension=".log", **save_kwargs):
    """
    log_file: target file
    operation: default '.copy'
    overwrite: default False
    """
    assert isinstance(log_dict, dict)
    log_dst = save_options(log_file, extension, **save_kwargs)
    with open(log_dst, "w") as f:
        json.dump(log_dict, f)
    print(f"processing log saved to \n {log_dst}")
