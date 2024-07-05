import time
import torch
import numpy as np

from utils.config import args
from utils.common import *
from utils.calculator import *
from utils.dataloader import get_loader, T
from utils.metrics import dice_func


def quantify():
    det_model = get_det_model(args, pretrained=True)
    seg_model = get_seg_model(args, pretrained=True)
    loader = get_loader(args)
    ds_name = args.data_root.split("/")[-1]
    res_writer = LogWriter(save_dir=args.log_save_dir, 
                           prefix=f"{ds_name}_{args.det_model}→{args.seg_model}")
    log_writer = LogWriter(save_dir=args.log_save_dir, 
                           prefix=f"Log_{ds_name}_{args.det_model}→{args.seg_model}")

    det_accs, seg_accs = [], []
    start_time = time.time()
    det_model.eval()
    seg_model.eval()
    for step, (image, mask, info) in enumerate(loader):
        ep_start = time.time()
        # load original image and mask
        mask_ = T.load_as_array(info["label"][0])   # original mask
        image_ = T.load_as_array(info["image"][0])  # original image
        max_nid = int(mask_.max() + 1)              # max nodule ID + 1
        # compute characteristics of original CT image:
        nodules_gt = {}
        for nid in range(1, max_nid):
            # get nodule and bounding box (ground truth):
            img = np.where(mask_ == nid, image_, 0)
            msk = np.where(mask_ == nid, 1, 0)
            # compute characteristics:
            volume_gt = calc_volume(msk)
            area_gt = calc_surface_area(msk)
            nodules_gt[nid] = {
                "volume_gt": round(volume_gt, 2),
                "surf_area_gt": round(area_gt, 2),
                "mass_gt": round(calc_mass(img, msk), 2),
                "sphericity_gt": round(calc_sphericity(volume_gt, area_gt), 2),
                "compactness_gt": round(calc_compactness(volume_gt, area_gt), 2),
                "elongation_gt": round(calc_elongation(volume_gt, area_gt), 2),
            }
        # model inference:
        with torch.no_grad():
            roi_df = detect_roi(det_model, image, args)
            rets = segment_nodules(seg_model, image, mask, image_, 
                                   nodules_gt, roi_df, args)
            det_accs.append(rets["det_acc"])
            seg_accs.append(rets["seg_acc"])
        # record accuracy:
        log_writer.add_row({
            "suid":     info["suid"][0],
            "det_acc":  round(rets["det_acc"], 4),  
            "seg_acc":  round(rets["seg_acc"], 4),
        })
        log_writer.save()
        # record characteristics:
        for line in rets["nodules"]:
            line["suid"] = info["suid"][0]
            res_writer.add_row(line)
        res_writer.save()
        print("Steps: %03d, ROI Num: %d," % (step, len(roi_df)),
              "Det Acc: %.5f," % rets["det_acc"],
              "Seg Acc: %.5f," % rets["seg_acc"],
              "Time: %.2fs" % (time.time()-ep_start))
        del image, mask, info
        torch.cuda.empty_cache()
    print("Mean Det Acc: %.5f," % np.mean(det_accs),
          "Mean Seg Acc: %.5f," % np.mean(seg_accs),
          "Total Time: %.2fmin" % ((time.time()-start_time) / 60))


def segment_nodules(seg_model, image, mask, img_arr, nodules_gt, roi_df, args):
    nodules = []
    # 获取ROI和图像的大小：
    roi_size = [args.roi_z, args.roi_x, args.roi_y]
    _, _, *shape = image.shape
    seg_result = np.zeros(list(shape))
    n_classes, det_accs = args.out_channels, []
    for idx, row in roi_df.iterrows():
        # 获取ROI中心点的坐标 以及 ROI的上下界范围：
        coords = [int(row["coord_z"]), int(row["coord_x"]),
                  int(row["coord_y"])]
        low = [max(int(c-r/2), 0) for c, r in zip(coords, roi_size)]
        high = [min(l+r, s) for l, r, s in zip(low, roi_size, shape)]
        low = [int(h-r) for h, r in zip(high, roi_size)]
        # 截取模型输入图像以及对应的掩膜图像：
        input = image[:, :, low[0]: high[0], low[1]: high[1], low[2]: high[2]]
        msk = mask[:, :, low[0]: high[0], low[1]: high[1], low[2]: high[2]]
        input, msk = input.to(args.device), msk.to(args.device)
        # 启动分割模型：
        seg_pred = seg_model(input).to(args.device)
        seg_out = seg_pred.argmax(dim=1).squeeze(dim=0).float().cpu().numpy()
        seg_result[low[0]: high[0], low[1]: high[1], low[2]: high[2]] = seg_out
        # 判断是否是真阳性：
        nid = torch.max(msk).item()
        det_accs.append(int(nid > 0))
        if nid > 0:
            if nid not in nodules_gt: continue
            nodule_gt = nodules_gt.pop(nid)
        else:
            nodule_gt = {
                "volume_gt": 0, "surf_area_gt": 0,
                "mass_gt": 0,   "sphericity_gt": 0,
                "compactness_gt": 0, "elongation_gt": 0,
            }
        img = img_arr[low[0]: high[0], low[1]: high[1], low[2]: high[2]]
        # 计算结节特征:
        volume_pd = calc_volume(seg_out)
        area_pd = calc_surface_area(seg_out)
        nodule_pd = {
            "nid": nid,
            "volume_pd": round(volume_pd, 2),
            "surf_area_pd": round(area_pd, 2),
            "mass_pd": round(calc_mass(img, seg_out), 2),
            "sphericity_pd": round(calc_sphericity(volume_pd, area_pd), 2),
            "compactness_pd": round(calc_compactness(volume_pd, area_pd), 2),
            "elongation_pd": round(calc_elongation(volume_pd, area_pd), 2),
        }
        # 合并nodule_gt和nodule_pd形成一条记录
        for key, val in nodule_gt.items():
            nodule_pd[key] = val
        nodules.append(nodule_pd)
    
    # 将nodules_gt中还没被合并的记录合并到nodules:
    for nid, nodule_gt in nodules_gt.items():
        nodule_pd = {
            "nid": nid,
            "volume_pd": 0, "surf_area_pd": 0,
            "mass_pd": 0, "sphericity_pd": 0,
            "compactness_pd": 0, "elongation_pd": 0,
        }
        # 合并nodule_gt和nodule_pd形成一条记录
        for key, val in nodule_gt.items():
            nodule_pd[key] = val
        nodules.append(nodule_pd)
        
    # compute accuracy
    mask = mask.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
    mask = np.where(mask > 0, 1, 0)
    seg_accs = dice_func(seg_result, mask, n_classes, ignore_bg=False)
    avg_seg_acc = np.mean(seg_accs) if len(seg_accs) > 0 else 1-np.min(mask)
    avg_det_acc = np.mean(det_accs) if len(det_accs) > 0 else 1-np.min(mask)
    return {
        "det_acc": avg_det_acc, 
        "seg_acc": avg_seg_acc, 
        "nodules": nodules,
    }


if __name__ == "__main__":
    quantify()
