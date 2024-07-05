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
    log_writer = LogWriter(save_dir=args.log_save_dir, 
                           prefix=f"MSD_{args.det_model}→{args.seg_model}")

    det_accs, seg_accs = [], []
    start_time = time.time()
    det_model.eval()
    seg_model.eval()
    for step, (image, mask, info) in enumerate(loader):
        ep_start = time.time()
        mask_ = T.load_as_array(info["label"][0])   # original mask
        origin = T.load_as_array(info["image"][0])  # original image
        true_volume = calc_volume(mask_)
        true_area = calc_surface_area(mask_)
        true_mass = calc_mass(origin, mask_)
        with torch.no_grad():
            roi_df = detect_roi(det_model, image, args)
            rets = det_based_segment(seg_model, image, mask,
                                     origin, roi_df, args)
        det_accs.append(rets["det_acc"])
        seg_accs.append(rets["seg_acc"])
        log_writer.add_row({
            "suid":     info["suid"][0],
            "det_acc":  round(rets["det_acc"], 4),
            "seg_acc":  round(rets["seg_acc"], 4),
            "true_volume":  round(true_volume, 2),
            "pred_volume":  round(rets["seg_volume"], 2),
            "true_mass":    round(true_mass, 2),
            "pred_mass":    round(rets["seg_mass"], 2),
            "true_area":    round(true_area, 2),
            "pred_area":    round(rets["seg_area"], 2),
        })
        log_writer.save()
        print("Steps: %03d, ROI Num: %d," % (step, len(roi_df)),
              "Det Acc: %.5f," % rets["det_acc"],
              "Seg Acc: %.5f," % rets["seg_acc"],
              "Time: %.2fs" % (time.time()-ep_start))
        del image, mask, info
        torch.cuda.empty_cache()
    print("Mean Det Acc: %.5f," % np.mean(det_accs),
          "Mean Seg Acc: %.5f," % np.mean(seg_accs),
          "Total Time: %.2fmin" % ((time.time()-start_time) / 60))


def det_based_segment(seg_model, image, mask, original, roi_df, args):
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
        det_accs.append(torch.max(msk).item())

    mask = mask.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
    seg_accs = dice_func(seg_result, mask, n_classes, ignore_bg=False)
    seg_volume = calc_volume(seg_result)
    seg_area = calc_surface_area(seg_result)
    seg_mass = calc_mass(original, seg_result)
    avg_det_acc = np.mean(det_accs) if len(det_accs) > 0 else 1-np.min(mask)
    return {
            "det_acc":      avg_det_acc,
            "seg_acc":      np.mean(seg_accs),
            "seg_volume":   seg_volume,     # pred volume
            "seg_area":     seg_area,       # pred surface area
            "seg_mass":     seg_mass,       # pred mass
        }


if __name__ == "__main__":
    quantify()
