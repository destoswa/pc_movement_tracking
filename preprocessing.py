import os
import numpy as np
import laspy
from omegaconf import OmegaConf
from src.format_conversions import convert_one_file


def preprocessing(conf):
    for file in [conf.data.src_pc1, conf.data.src_pc2]:
        if file.endswith('.copc.laz'):
            file_name = file.split('.copc.laz')[0]
            file_ext = 'copclaz'
        else:
            file_name, file_ext = os.path.splitext(file)
            file_ext = file_ext[1:]

        # Convert to LAZ
        if file_ext != 'laz':
            new_file = file_name + '.laz'
            convert_one_file(file, new_file, file_ext, 'laz')
            file = new_file

        # Remove classes
        pointCloud = laspy.read(file)
        mask = np.ones(len(pointCloud), dtype=np.bool_)
        Classification = getattr(pointCloud, "classification")
        for val in conf.preprocessing.list_cat_to_remove:
            mask[Classification == val] = False
        pointCloud.points = pointCloud.points[mask]
        pointCloud.write(file)

        # Convert to PLY
        convert_one_file(file, file_name + '.ply', 'laz', 'ply', do_keep_sf=conf.preprocessing.do_keep_sf)


if __name__ == "__main__":
    conf = OmegaConf.load('./config.yaml')
    preprocessing(conf)

