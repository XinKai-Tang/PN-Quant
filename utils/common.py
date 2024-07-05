import os
import time
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk

from argparse import ArgumentParser
from math import ceil, floor
from typing import Union, Sequence
from torch.nn import Module, DataParallel, functional as F

from nets import UNETR, SwinUNETR, UXNET, UPerNeXt
from nets import NoduleNet, SANet, NNet, DeepLung, DeepSEED


def detect_roi(det_model: Module,
               image: torch.Tensor,
               args: ArgumentParser):
    ''' detect roi by detection model '''
    if isinstance(det_model, NoduleNet):
        image = pad_or_crop(image, 16)
        result = nodulenet_infer(det_model, image, args)
    elif isinstance(det_model, SANet):
        image = pad_or_crop(image, 32)
        result = nodulenet_infer(det_model, image, args)
    else:
        image = pad_or_crop(image, 16)
        result = deeplung_infer(det_model, image, args)
    torch.cuda.empty_cache()    # 清空GPU的Cache
    return result


def pad_or_crop(image: torch.Tensor,
                factor: int = 16):
    ''' pad or crop image to factor '''
    shape, crop_th = image.shape[-3:], 0 # int(factor * 0.6)
    # 计算裁剪或填充所需的最小大小：
    crops = [s - floor(s / factor) * factor for s in shape]
    pads = [ceil(s / factor) * factor - s for s in shape]
    # 为了防止填充或裁剪区域过大，我们必须要进行权衡：
    for idx, cs in enumerate(crops):
        if cs > crop_th:    # pad image to factor
            crops[idx] = 0
        else:               # crop image to factor
            pads[idx] = 0
    # crop image to factor
    shape = [s - c for s, c in zip(shape, crops)]
    image = image[..., :shape[0], :shape[1], :shape[2]]
    # pad image to factor
    pads = [0, pads[2], 0, pads[1], 0, pads[0]]
    image = F.pad(image, pads, 'constant', 0)
    return image


def nodulenet_infer(det_model: Module,
                    inputs: torch.Tensor,
                    args: ArgumentParser):
    ''' infer function of NoduleNet/SANet '''
    cols = ["prob", "coord_z", "coord_x", "coord_y",
             "bbox_z", "bbox_x", "bbox_y"]
    # 准备目标检测模型所需的相关参数：
    inputs = inputs.to(args.device)
    gt_bboxes = torch.tensor(data=[], device=args.device)
    gt_labels = torch.tensor(data=[[1]], device=args.device)
    gt_masks = torch.ones_like(inputs, device=args.device)
    masks = torch.ones_like(inputs, device=args.device)
    # 获取目标检测的结果：
    det_model(inputs, gt_bboxes, gt_labels, gt_masks, masks)
    esmb = det_model.ensemble_proposals.cpu().numpy()
    if len(esmb):
        result = pd.DataFrame(data=esmb[:, 1:], columns=cols)
        result = merge_roi(result, min_prob=args.min_prob,
                           roi_size=(args.roi_z, args.roi_x, args.roi_y))
    else:
        result = pd.DataFrame(columns=cols)
    del inputs, gt_bboxes, gt_labels, gt_masks, masks
    return result


def deeplung_infer(det_model: Module,
                   inputs: torch.Tensor,
                   args: ArgumentParser):
    ''' infer function of DeepLung/DeepSEED/NNet '''
    cols = ["prob", "coord_x", "coord_y", "coord_z",
             "bbox_x", "bbox_y", "bbox_z"]
    inputs = inputs.to(args.device)
    xx,yy,zz = np.meshgrid(*[np.linspace(-0.5, 0.5, size // det_model.cfg["stride"])
                             for size in inputs.shape[-3:]], indexing='ij')
    coords = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...], 
                             zz[np.newaxis,:]], 0, dtype="float32")
    coords = torch.tensor(coords[np.newaxis,...], device=args.device)
    # 获取目标检测的结果
    outputs = det_model(inputs, coords)
    pbbs = det_model.get_pbb(outputs[0].cpu().numpy(), -3)
    if len(pbbs) > 0:
        result = pd.DataFrame(pbbs[..., :-1], columns=cols[:4])
        result[cols[4]] = result[cols[5]] = result[cols[6]] = pbbs[..., -1]
        result = merge_roi(result, min_prob=args.min_prob,
                           roi_size=(args.roi_z, args.roi_x, args.roi_y))
    else:
        result = pd.DataFrame(columns=cols)
    del inputs, coords, outputs
    return result


def get_seg_model(args: ArgumentParser, pretrained: bool = True):
    ''' get segmentation model '''
    seg_model_name = args.seg_model.lower()
    if seg_model_name == "unetr":
        seg_model = UNETR(in_channels=args.in_channels,
                          out_channels=args.out_channels,
                          img_size=(args.roi_z, args.roi_x, args.roi_y))
    elif seg_model_name == "swinunetr":
        seg_model = SwinUNETR(in_channels=args.in_channels,
                              out_channels=args.out_channels)
    elif seg_model_name == "uxnet":
        seg_model = UXNET(in_channels=args.in_channels,
                          out_channels=args.out_channels)
    elif seg_model_name == "upernextv1":
        seg_model = UPerNeXt(in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             use_grn=False)
    elif seg_model_name == "upernextv2":
        seg_model = UPerNeXt(in_channels=args.in_channels,
                             out_channels=args.out_channels,
                             use_grn=True)
    else:
        raise ValueError(
            f"Segmentation Model `{args.seg_model}` is not supported!"
        )
    seg_model = DataParallel(seg_model).to(args.device)

    if pretrained:
        path = os.path.join(args.model_save_dir, args.seg_model, 
                            args.trained_seg_model)
        state_dict = torch.load(path)
        seg_model.load_state_dict(state_dict["net"])
        print("LOAD checkpoints from `%s`..." % path)
    return seg_model


def get_det_model(args: ArgumentParser, pretrained: bool = True):
    ''' get detection model '''
    det_model_name = args.det_model.lower()
    if det_model_name == "nodulenet":
        det_model = NoduleNet(mode="test")
        det_model.use_mask = False
        det_model.use_rcnn = True
    elif det_model_name == "sanet":
        det_model = SANet(mode="test")
        det_model.use_rcnn = True
    elif det_model_name == "nnet":
        det_model = NNet(mode="test")
    elif det_model_name == "deeplung":
        det_model = DeepLung(mode="test")
    elif det_model_name == "deepseed":
        det_model = DeepSEED(mode="test")
    else:
        raise ValueError(
            f"Detection Model `{args.det_model}` is not supported!"
        )
    det_model = det_model.to(args.device)

    if pretrained:
        path = os.path.join(args.model_save_dir, args.det_model,
                            args.trained_det_model)
        state_dict = torch.load(path, map_location=args.device)
        det_model.load_state_dict(state_dict["state_dict"])
        print("LOAD checkpoints from `%s`..." % path)
    return det_model


def merge_roi(result: pd.DataFrame,
              min_prob: float = 0.5,
              roi_size: tuple = (64, 64, 64)):
    ''' merge near ROI. '''
    threshold = min(roi_size) ** 2      # 设置最小平方欧式距离
    det_df = result[result["prob"] > min_prob].copy()
    det_df.sort_values(by=["prob"], ascending=False, inplace=True)
    det_df.reset_index(drop=True, inplace=True)     # 更新index
    i, j = 0, 1
    while i + 1 < det_df.shape[0]:
        while j < det_df.shape[0]:
            # 计算第i、j个ROI最远端顶点的X坐标：
            if det_df.iloc[i]["coord_x"] < det_df.iloc[j]["coord_x"]:
                x1 = det_df.iloc[i]["coord_x"] - det_df.iloc[i]["bbox_x"] / 2
                x2 = det_df.iloc[j]["coord_x"] + det_df.iloc[j]["bbox_x"] / 2
            else:
                x1 = det_df.iloc[i]["coord_x"] + det_df.iloc[i]["bbox_x"] / 2
                x2 = det_df.iloc[j]["coord_x"] - det_df.iloc[j]["bbox_x"] / 2
            # 计算第i、j个ROI最远端顶点的Y坐标：
            if det_df.iloc[i]["coord_y"] < det_df.iloc[j]["coord_y"]:
                y1 = det_df.iloc[i]["coord_y"] - det_df.iloc[i]["bbox_y"] / 2
                y2 = det_df.iloc[j]["coord_y"] + det_df.iloc[j]["bbox_y"] / 2
            else:
                y1 = det_df.iloc[i]["coord_y"] + det_df.iloc[i]["bbox_y"] / 2
                y2 = det_df.iloc[j]["coord_y"] - det_df.iloc[j]["bbox_y"] / 2
            # 计算第i、j个ROI最远端顶点的Z坐标：
            if det_df.iloc[i]["coord_z"] < det_df.iloc[j]["coord_z"]:
                z1 = det_df.iloc[i]["coord_z"] - det_df.iloc[i]["bbox_z"] / 2
                z2 = det_df.iloc[j]["coord_z"] + det_df.iloc[j]["bbox_z"] / 2
            else:
                z1 = det_df.iloc[i]["coord_z"] + det_df.iloc[i]["bbox_z"] / 2
                z2 = det_df.iloc[j]["coord_z"] - det_df.iloc[j]["bbox_z"] / 2
            # 计算这两个顶点的平方欧氏距离：
            dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
            dist = dx ** 2 + dy ** 2 + dz ** 2
            if dist < threshold:
                # 计算这两个顶点的中心点坐标：
                cx, cy, cz = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
                p = max(det_df.iloc[i]["prob"], det_df.iloc[j]["prob"])
                # 对于距离较近的ROI需要进行合并：
                det_df.iloc[i] = [p, cz, cx, cy, dz, dx, dy]
                det_df.drop(j, axis=0, inplace=True)
            j += 1
        det_df.reset_index(drop=True, inplace=True)     # 更新index
        i, j = i + 1, i + 2
    return det_df


def save_segmentation(result: Union[np.ndarray, torch.Tensor],
                      args: ArgumentParser,
                      axis: int = 1,
                      save_name: str = "segmentation.npy"):
    ''' save segmentation result as a file '''
    # get the save path of the segmentation result:
    path = os.path.join(args.segmentation_dir, args.model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, save_name)
    # post-process the segmentation result:
    if isinstance(result, torch.Tensor):
        result = torch.softmax(result, axis).detach().cpu().numpy()
    if result.shape[axis] > 1:
        result = np.argmax(result, axis).astype(np.uint8)[0]
    # save the quantification result to a file:
    if path[-4:] == ".npy":
        np.save(path, result)
    else:
        res_img = sitk.GetImageFromArray(result)
        sitk.WriteImage(res_img, path)
    print("SAVE segmentation to `%s`..." % path)


class LogWriter:
    ''' Log Writer Based on Pandas '''

    def __init__(self, save_dir: str, prefix: str = None):
        ''' Args:
        * `save_dir`: save place of log files.
        * `prefix`: prefix-name of the log file.
        '''
        self.data = pd.DataFrame()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        now = time.strftime("%y%m%d%H%M", time.localtime())
        fname = f"{prefix}-{now}.csv" if prefix else f"{now}.csv"
        self.path = os.path.join(save_dir, fname)

    def add_row(self, data: dict):
        temp = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, temp], ignore_index=True)

    def save(self):
        self.data.to_csv(self.path, index=False)
        print("SAVE runtime logs to `%s`..." % self.path)
