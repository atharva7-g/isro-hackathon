import numpy as np
import re
from collections import defaultdict
import rasterio
from pathlib import Path
import os
from glob import glob
from rasterio import errors
from rasterio.transform import Affine
from rasterio.enums import Resampling

class utils:
    def __init__(self):
        pass

    @staticmethod
    def check(dir):
        unreadable_files = []
        
        for filepath in glob(os.path.join(dir, "*.tif")):
            try:
                with rasterio.open(filepath) as src:
                    pass
            except errors.RasterioIOError:
                unreadable_files.append(filepath)
        
        if unreadable_files:
            unreadable_files.sort()
            print("Unreadable files:")
            for f in unreadable_files:
                print(f)
        else:
            print("All files are readable.")



class Preprocessing:
    def __init__(self, input_len = 4, pred_len = 1, output_dir = "stacked/", input_dir = "images/INSAT3DR-L1C", band_order = ['VIS', 'SWIR', 'MIR', 'TIR1', 'TIR2', 'WV']):
        self.band_order = band_order
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_len = input_len
        self.pred_len = pred_len

    def normalize(self, array):
        min, max = np.min(array), np.max(array)
        if max > min:
            return (array - min) / (max - min) 
        else:
            return array * 0    

    def attach_desc(self):
        pass

    def resize(self, output_folder='resized/', scale_factor=0.25):
        input_folder = self.output_dir
        scale_factor = 0.25

        for file in glob(os.path.join(input_folder, "*.tif")):
            with rasterio.open(file) as src:
                width = int(src.width * scale_factor)
                height = int(src.height * scale_factor)

                new_res_x = (src.bounds.right - src.bounds.left) / width
                new_res_y = (src.bounds.top - src.bounds.bottom) / height
                new_transform = Affine.translation(src.bounds.left, src.bounds.top) * Affine.scale(new_res_x, -new_res_y)

                kwargs = src.meta.copy()
                kwargs.update({
                    'transform': new_transform,
                    'width': width,
                    'height': height
                })

                dst_path = os.path.join(output_folder, os.path.basename(file))

                with rasterio.open(dst_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        src_data = src.read(
                            i,
                            out_shape=(height, width),
                            resampling=Resampling.bilinear
                        )
                        dst.write(src_data, i)
    
    def get_stacked_raster_filenames(self, directory='resized/', pattern='_stack.tif'):
        directory = Path(directory)
    
        files = sorted([f for f in directory.iterdir() 
                        if f.is_file() and f.name.endswith(pattern)])
    
        if not files:
            print(f"No files found matching pattern '{pattern}' in {directory}")
            return []
    
        print(f"Found {len(files)} raster files")
        for f in files:
            print(f"File: {f.name}")
    
        return files

    def sequentiate(self, stacked_arrays):        
        T = stacked_arrays.shape[0]
        N = T - self.input_len - self.pred_len + 1
        if N <= 0:
            raise ValueError("Not enough time steps for the given input_len and pred_len")
        
        X = []
        y = []
        for i in range(N):
            X.append(stacked_arrays[i : i + self.input_len])
            y.append(stacked_arrays[i + self.input_len : i + self.input_len + self.pred_len])
        return np.array(X), np.array(y)


    def stack(self):
        pattern = re.compile(r"3DIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_.*_IMG_(\w+)\.tif")
        grouped_files = defaultdict(dict)
        
        band_paths = os.path.join(self.input_dir, "*.tif")

        for filepath in glob(band_paths):
            match = pattern.search(os.path.basename(filepath))
            if match:
                date, time, band = match.groups()
                timestamp = f"{date}_{time}"
                grouped_files[timestamp][band] = filepath
        
        for timestamp, files_dict in grouped_files.items():
            if all(band in files_dict for band in self.band_order):
                stack = []
                meta = None
                for band in self.band_order:
                    with rasterio.open(files_dict[band]) as src:
                        data = src.read(1)
                        stack.append(data)
                        if meta is None:
                            meta = src.meta.copy()
                stack_array = np.stack(stack)
                meta.update(count=len(self.band_order))
                output_path = os.path.join(self.output_dir, f"{timestamp}_stack.tif")
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(stack_array)
                print(f"Saved: {output_path}")
            else:
                print(f"Skipping {timestamp}: incomplete band set")

        utils.check(self.output_dir)
 