import math
import numpy as np
import pandas as pd
import SimpleITK as sitk

from datetime import datetime
from typing import Tuple, Union


def count_voxels(mask: Union[list, np.ndarray]) -> int:
    ''' count voxels of target zone '''
    mask = np.array(mask > 0, dtype=np.int16)
    return mask.sum()


def calc_intensity(image: Union[list, np.ndarray],
                   mask: Union[list, np.ndarray]) -> Tuple[float, float]:
    ''' calculate intensity of target zone '''
    num = count_voxels(mask)
    if num == 0:
        avg_gray, std_gray = 0.0, 0.0
    else:
        image = np.array(image, dtype=np.int16)
        mask = np.array(mask, dtype=np.int16)
        seg = np.multiply(image, mask)
        avg_gray = seg.sum() / num
        std_gray = seg.std()
    return avg_gray, std_gray


def calc_mass(image: Union[list, np.ndarray],
              mask: Union[list, np.ndarray],
              spacing: tuple = (1., 1., 1.)) -> float:
    ''' calculate mass of target zone '''
    volume = calc_volume(mask, spacing)
    avg_gray, _ = calc_intensity(image, mask)
    mass = volume * (avg_gray + 1000) / 1000
    return mass


def calc_diameter(mask: Union[list, np.ndarray],
                  spacing: tuple = (1., 1., 1.)) -> float:
    ''' calculate mean diameter of target zone '''
    z_gt, y_gt, x_gt = np.where(mask > 0)
    if z_gt.shape[0] == 0:
        diameter = 0.0
    else:
        d_z = (z_gt.max() - z_gt.min() + 1) * spacing[0]
        d_y = (y_gt.max() - y_gt.min() + 1) * spacing[1]
        d_x = (x_gt.max() - x_gt.min() + 1) * spacing[2]
        diameter = np.mean([d_z, d_y, d_x])
    return diameter


def calc_volume(mask: Union[list, np.ndarray],
                spacing: tuple = (1., 1., 1.)) -> float:
    ''' calculate volume of target zone '''
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume = count_voxels(mask) * voxel_volume
    return volume


def calc_surface_area(mask: Union[list, np.ndarray],
                      spacing: tuple = (1., 1., 1.)) -> float:
    ''' calculate surface area of target zone '''
    mask = np.array(mask)
    if mask.ndim == 3:    # mask: (D, H, W)
        mask = mask[np.newaxis, ...]
    # y-O-z surface:
    A1 = np.roll(mask, shift=1, axis=1)
    A1 = np.logical_xor(A1, mask)
    s1 = A1.sum() * spacing[1] * spacing[2]
    # x-O-z surface:
    A2 = np.roll(mask, shift=1, axis=2)
    A2 = np.logical_xor(A2, mask)
    s2 = A2.sum() * spacing[0] * spacing[2]
    # x-O-y surface:
    A3 = np.roll(mask, shift=1, axis=3)
    A3 = np.logical_xor(A3, mask)
    s3 = A3.sum() * spacing[0] * spacing[1]
    return s1 + s2 + s3


def calc_dbl_time(vol1: float, vol2: float,
                  date1: str, date2: str,
                  date_format: str = r"%Y%m%d") -> float:
    ''' calculate mean doubling time '''
    if vol1 > vol2:
        print("Warning: `vol2` is smaller than `vol1`.")
    if date1 >= date2:
        raise ValueError("`date2` must be later than `date1`.")
    date1 = datetime.strptime(date1, date_format)
    date2 = datetime.strptime(date2, date_format)
    n_days = (date2 - date1).days
    delta = math.log(vol2) - math.log(vol1)
    dbl_time = math.log(2) * n_days / delta
    return dbl_time


def calc_sphericity(volume: float, area: float) -> float:
    ''' calculate sphericity of target zone '''
    if area == 0: return 1
    sph = 36 * math.pi * (volume ** 2)
    sph = pow(sph, 1/3) / area
    return sph


def calc_compactness(volume: float, area: float) -> float:
    ''' calculate compactness of target zone '''
    if area == 0: return 1
    cmp = pow(math.pi, 1/2) * pow(area, 2/3)
    cmp = volume / cmp
    return cmp


def calc_elongation(volume: float, area: float) -> float:
    ''' calculate elongation of target zone '''
    if area == 0: return 1
    rad = pow(3 / 4 / math.pi * volume, 1/3)
    elong = area / (4 * math.pi * rad * rad)
    return elong


class QuantMethod:
    def __init__(self,
                 img_path: str,
                 patient_id: str,
                 study_date: str):
        ''' Args:
        * `img_path`: original ct image path.
        * `patient_id`: id number of the patient.
        * `study_date`: study date of the ct.
        '''
        self.img_path = img_path
        self.patient_id = patient_id
        self.study_date = study_date

    def __call__(self,
                 image: Union[list, np.ndarray] = None,
                 result: Union[list, np.ndarray] = None,
                 spacing: tuple = (1, 1, 1),
                 decimal: int = 3):
        if image is None or result is None:
            volume = area = avg_ct = std_ct = mass = 0
            sphericity = compactness = elongation = 1
        else:
            volume = calc_volume(result, spacing)
            area = calc_surface_area(result, spacing)
            avg_ct, std_ct = calc_intensity(image, result)
            mass = volume * (avg_ct + 1000) / 1000
            sphericity = calc_sphericity(volume, area)
            compactness = calc_compactness(volume, area)
            elongation = calc_elongation(volume, area)
        quant_df = pd.DataFrame({
            "pid": self.patient_id,
            "date": self.study_date,
            "avg_hu": round(avg_ct, decimal),
            "std_hu": round(std_ct, decimal),
            "mass": round(mass, decimal),
            "volume": round(volume, decimal),
            "area": round(area, decimal),
            "sphericity": round(sphericity, decimal),
            "compactness": round(compactness, decimal),
            "elongation": round(elongation, decimal),
        }, index=[0])
        return quant_df

    def get_original_image(self):
        ''' load ct image and its spacing '''
        if self.img_path[-4:] == ".npy":
            dcm_arr = np.load(self.img_path)
            spacing = (1., 1., 1.)
        else:
            dcm_img = sitk.ReadImage(self.img_path)
            dcm_arr = sitk.GetArrayFromImage(dcm_img)
            spacing = list(dcm_img.GetSpacing())
        return dcm_arr, spacing
