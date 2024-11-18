import json
from pathlib import Path
import re
import tifffile
from nd2reader import ND2Reader


def load_custom_config(path='custom_config.json'):
	with open(path) as f:
		custom_config = json.load(f)
	return custom_config


def check_extenstion(filename, extension='.tif'):
    if not isinstance(filename, Path):
        filename = Path(filename)
    return filename.suffix == extension


def read_tiff(filename):
    img = tifffile.imread(filename)
    #img = np.moveaxis(img, 0, -1)
    return img 


def read_nd2(filename, bundle_axes='zcyx'):
    with ND2Reader(filename) as images:
        if bundle_axes is not None:
            images.bundle_axes = bundle_axes
            images = images[0]
            return images
        return images


def read_img(filename, do_3D=False):
    if not isinstance(filename, (str, Path)):
        raise TypeError(f'filename should be of type str of pathlib.Path, but is {type(filename)}.')
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.suffix == '.tif':
        return tifffile.imread(str(filename))
    elif filename.suffix == '.nd2':
        bundle_axes = 'zcyx' if do_3D else 'cyx'
        return read_nd2(str(filename), bundle_axes=bundle_axes)


def read_2mirror_img(filename):
    img = tifffile.imread(filename)
    _, Z, H, W = img.shape
    img = img.reshape(img, (2*Z, H, W))
    return img[0::2,], img[1::2,]


def get_files_in_folder(path, file_extension):
	if not isinstance(path, Path):
		path = Path(path)
	files = sorted([f for f in path.iterdir() if f.is_file() and \
        f.suffix == file_extension and not str(f.name).startswith('.')])
	return files


def get_subfolders(path):
	if not isinstance(path, Path):
		path = Path(path)
	folders = [f for f in path.iterdir() if f.is_dir()]
	return folders


def extract_cell_position_from_file_name(path):
        file_name = Path(path).stem
        position = [int(s) for s in re.split('X|Y|Z|-',file_name) if s.isdigit()]
        return position