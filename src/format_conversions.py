import os
import sys
import numpy as np
from tqdm import tqdm
import json
import laspy
import numpy as np
import open3d as o3d
import traceback
from plyfile import PlyData, PlyElement

class Convertions:
    @staticmethod
    def convert_laz_to_las(in_laz, out_las, verbose=True, **kwargs):
        """
        Converts a LAZ file to an uncompressed LAS file.

        Args:
            - in_laz (str): Path to the input .laz file.
            - out_las (str): Path to the output .las file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .las file.
        """

        las = laspy.read(in_laz)
        las = laspy.convert(las)
        las.write(out_las)
        if verbose:
            print(f"LAS file saved in {out_las}")

    @staticmethod
    def convert_las_to_laz(in_las, out_laz, verbose=True, **kwargs):
        """
        Converts a LAZ file to an uncompressed LAS file.

        Args:
            - in_laz (str): Path to the input .laz file.
            - out_las (str): Path to the output .las file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .las file.
        """

        las = laspy.read(in_las)
        las = laspy.convert(las)
        las.write(out_laz)
        if verbose:
            print(f"LAS file saved in {out_laz}")

    @staticmethod
    def convert_pcd_to_laz(in_pcd, out_laz, verbose=True, **kwargs):
        """
        Converts a PCD file to a compressed LAZ file using PDAL.

        Args:
            - in_pcd (str): Path to the input .pcd file.
            - out_laz (str): Path to the output .laz file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .laz file after reprojecting to EPSG:2056.
        """

        # pcd = laspy.read('../data/testing_samples/split_0332.pcd')
        pipeline_json = {
            "pipeline": [
                in_pcd,  # Read the PCD file
                {
                    "type": "writers.las",
                    "filename": out_laz,
                    "compression": "laszip"  # Ensures .laz compression
                    ""
                },
                {
                    "type": "filters.reprojection",
                    "in_srs": "EPSG:4326",
                    "out_srs": "EPSG:2056"
                }
            ]
        }

        # Run the PDAL pipeline
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        
        if verbose:
            print(f"LAZ file saved in {out_laz}")

    @staticmethod
    def convert_laz_to_pcd(in_laz, out_pcd, verbose=True, **kwargs):
        """
        Converts a LAZ file to a PCD file, preserving all point attributes.

        Args:
            - in_laz (str): Path to the input .laz file.
            - out_pcd (str): Path to the output .pcd file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .pcd file in ASCII format.
        """

        laz = laspy.read(in_laz)

        # Gathering all attributes from laz file
        points = np.vstack((laz.x, laz.y, laz.z)).T

        attributes = {}
        for attribute in laz.point_format.dimensions:
            if attribute.name in ['X', 'Y', 'Z']:
                continue
            attributes[attribute.name] = getattr(laz, attribute.name)
        
        # Preparing data for pcd
        num_points = points.shape[0]
        fields = ["x", "y", "z"] + list(attributes.keys())  # All field names
        types = ["F", "F", "F"] + ["F" for _ in attributes]  # Float32 fields
        sizes = [4] * len(fields)  # 4-byte float per field

        # Stack all data into a single NumPy array
        data = np.column_stack([points] + [attributes[key] for key in attributes])

        # Write to a PCD file
        with open(out_pcd, "w") as f:
            # f.write(f"# .PCD v0.7 - Point Cloud Data file format\n")
            f.write(f"VERSION 0.7\n")
            f.write(f"FIELDS {' '.join(fields)}\n")
            f.write(f"SIZE {' '.join(map(str, sizes))}\n")
            f.write(f"TYPE {' '.join(types)}\n")
            f.write(f"COUNT {' '.join(['1'] * len(fields))}\n")
            f.write(f"WIDTH {num_points}\n")
            f.write(f"HEIGHT 1\n")
            f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write(f"DATA ascii\n")
        
            # Write data
            np.savetxt(f, data, fmt=" ".join(["%.6f"] * len(fields)))
        f.close()
        if verbose:
            print(f"PCD file saved in {out_pcd}")

    @staticmethod
    def convert_las_to_ply(input_path, output_path, use_color=True, do_keep_sf=False, verbose=False, **kwargs):
        las = laspy.read(input_path)

        x = np.array(las.x, dtype=np.float32)
        y = np.array(las.y, dtype=np.float32)
        z = np.array(las.z, dtype=np.float32)

        has_color = use_color and all(
            c in las.point_format.dimension_names for c in ("red", "green", "blue")
        )

        # Fields to skip (handled separately)
        skip_fields = {"x", "y", "z", "X", "Y", "Z"}
        if has_color:
            skip_fields.update({"red", "green", "blue"})

        scalar_fields = []
        if do_keep_sf:
            # Collect scalar fields
            for name in las.point_format.dimension_names:
                if name in skip_fields:
                    continue
                
                values = np.array(getattr(las, name))
                dtype = values.dtype
                scalar_fields.append((f"scalar_{name}", values, dtype))

        # Build dtype and data
        dtypes = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
        arrays = [x, y, z]

        if has_color:
            r = (las.red / 65535.0 * 255).astype(np.uint8)
            g = (las.green / 65535.0 * 255).astype(np.uint8)
            b = (las.blue / 65535.0 * 255).astype(np.uint8)
            dtype += [("red", np.uint8), ("green", np.uint8), ("blue", np.uint8)]
            arrays += [r, g, b]

        for name, values, dtype in scalar_fields:
            dtypes.append((name, dtype))
            arrays.append(values)

        vertex_data = np.array(list(zip(*arrays)), dtype=dtypes)

        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el], text=False).write(output_path)

        if verbose:
            scalar_names = [n for n, _ in scalar_fields]
            print(f"Saved to {output_path}")
            print(f"  Points   : {len(x)}")
            print(f"  Color    : {has_color}")
            print(f"  Scalars  : {scalar_names}")
    
    @staticmethod
    def convert_laz_to_ply(input_path, output_path, use_color=True, do_keep_sf=False, verbose=False, **kwargs):
        las = laspy.read(input_path)

        x = np.array(las.x, dtype=np.float32)
        y = np.array(las.y, dtype=np.float32)
        z = np.array(las.z, dtype=np.float32)

        has_color = use_color and all(
            c in las.point_format.dimension_names for c in ("red", "green", "blue")
        )

        # Fields to skip (handled separately)
        skip_fields = {"x", "y", "z", "X", "Y", "Z"}
        if has_color:
            skip_fields.update({"red", "green", "blue"})

        scalar_fields = []
        if do_keep_sf:
            # Collect scalar fields
            for name in las.point_format.dimension_names:
                if name in skip_fields:
                    continue
                
                values = np.array(getattr(las, name))
                dtype = values.dtype
                scalar_fields.append((f"scalar_{name}", values, dtype))

        # Build dtype and data
        dtypes = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
        arrays = [x, y, z]

        if has_color:
            r = (las.red / 65535.0 * 255).astype(np.uint8)
            g = (las.green / 65535.0 * 255).astype(np.uint8)
            b = (las.blue / 65535.0 * 255).astype(np.uint8)
            dtype += [("red", np.uint8), ("green", np.uint8), ("blue", np.uint8)]
            arrays += [r, g, b]

        for name, values, dtype in scalar_fields:
            dtypes.append((name, dtype))
            arrays.append(values)

        vertex_data = np.array(list(zip(*arrays)), dtype=dtypes)

        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el], text=False).write(output_path)

        if verbose:
            scalar_names = [n for n, _ in scalar_fields]
            print(f"Saved to {output_path}")
            print(f"  Points   : {len(x)}")
            print(f"  Color    : {has_color}")
            print(f"  Scalars  : {scalar_names}")

    @staticmethod
    def convert_copclaz_to_laz(src_in, src_out, verbose=True, **kwargs):
        las = laspy.read(src_in)
        points = las.points.copy()
        new_header = laspy.LasHeader(
                    version=las.header.version,
                    point_format=las.header.point_format
        )
        new_header.offsets = las.header.offsets
        new_header.scales = las.header.scales
        with laspy.open(src_out, mode="w", header=new_header) as f:
            f.write_points(points)
        if verbose:
            print(f"COPCLAZ file saved in {src_out}")


def convert_all_in_folder(src_folder_in, src_folder_out, in_type, out_type, verbose=False, **kwargs):
    """
    Converts all files in a folder from one point cloud format to another.

    Args:
        - src_folder_in (str): Path to the input folder containing files to convert.
        - src_folder_out (str): Path to the output folder where converted files will be saved.
        - in_type (str): Input file type ('las', 'laz', or 'pcd').
        - out_type (str): Output file type ('las', 'laz', or 'pcd').
        - verbose (bool, optional): Whether to display a progress bar and detailed messages. Defaults to False.

    Returns:
        - None: Saves all converted files into the specified output folder.
    """
    
    assert in_type in ['copclaz', 'las', 'laz', 'pcd']
    assert out_type in ['las', 'laz', 'pcd', 'ply']
    assert in_type != out_type

    if not hasattr(Convertions, f"convert_{in_type}_to_{out_type}"):
        print(f"No function for converting {in_type} into {out_type}!!")
        return
    os.makedirs(src_folder_out, exist_ok=True)  # Ensure output folder exists
    files = [f for f in os.listdir(src_folder_in) if f.endswith('.' + in_type)]
    for _, file in tqdm(enumerate(files), total=len(files), desc=f"Converting {in_type} in {out_type}", disable=~verbose):
        try:
            file_out = file.split(in_type)[0] + out_type
            _ = getattr(Convertions, f"convert_{in_type}_to_{out_type}")(os.path.join(src_folder_in, file), os.path.join(src_folder_out, file_out), verbose=verbose)
        except Exception as e:
            print(f"conversion from {in_type} to {out_type} for sample {file} failed")
            print('error: ', e)
            pass


def convert_one_file(src_file_in, src_file_out, in_type, out_type, offsets=[0,0,0], **kwargs):
    assert in_type in ['copclaz', 'las', 'laz', 'txt']
    assert out_type in ['las', 'laz', 'txt', 'ply']
    assert in_type != out_type

    if not hasattr(Convertions, f"convert_{in_type}_to_{out_type}"):
        print(f"No function for converting {in_type} into {out_type}!!")
        return
    try:
        _ = getattr(Convertions, f"convert_{in_type}_to_{out_type}")(src_file_in, src_file_out, verbose=False, **kwargs)
    except Exception as e:
        print(f"conversion from {in_type} to {out_type} for sample {src_file_in} failed")
        print(traceback.format_exc())
        pass


if __name__ == "__main__":
    src_folder_in = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_10_swam_and_lake\2588_1168_lake\results\new"
    src_folder_out = src_folder_in
    in_type = 'las'
    out_type = 'ply'
    convert_all_in_folder(src_folder_in, src_folder_out, in_type, out_type, True)


    # if len(sys.argv) >= 5:
    #     src_folder_in = sys.argv[1]
    #     src_folder_out = sys.argv[2]
    #     in_type = sys.argv[3]
    #     out_type = sys.argv[4]
    #     verbose = False
    #     if len(sys.argv) == 6:
    #         if sys.argv[5].lower() == "true":
    #             verbose = True
        
    #     convert_all_in_folder(src_folder_in, src_folder_out, in_type, out_type, verbose)
    # else:
    #     print("Missing arguments!")
    #     quit()