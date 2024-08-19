import os
import setproctitle
from argparse import ArgumentParser
from torch import cuda, device as Device

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
setproctitle.setproctitle("txk-NoduleQua")

parser = ArgumentParser(description="Image Quantification of Lung Nodules")
parser.add_argument("--det_model", type=str, default="NoduleNet", 
                    help="detection model (eg: SANet, NoduleNet)")
parser.add_argument("--seg_model", type=str, default="UXNet", 
                    help="segmentation model (eg: UXNet, SwinUNETR)")
parser.add_argument("--device", type=Device, help="runtime device",
                    default=Device("cuda" if cuda.is_available() else "cpu"))

############################ save path ############################
parser.add_argument("--data_root", type=str, default="dataset",
                    help="save path of test dataset")
parser.add_argument("--image_dir", type=str, default="images",
                    help="name of image filefloder")
parser.add_argument("--mask_dir", type=str, default="masks",
                    help="name of masks filefolder")

# best: NoduleNet-2  SANet-2  SwinUNETR-1  UXNet-0
parser.add_argument("--model_save_dir", type=str, default="models",
                    help="save path of trained models")
parser.add_argument("--trained_det_model", type=str, default="NoduleNet-2.ckpt",
                    help="filename of pretrained detection model")
parser.add_argument("--trained_seg_model", type=str, default="best_model_f0.pth",
                    help="filename of pretrained segmentation model")

parser.add_argument("--log_save_dir", type=str, default="logs",
                    help="save path of runtime logs")
parser.add_argument("--segmentation_dir", type=str, default="segs",
                    help="filename of segmentation results")
parser.add_argument("--quantification_dir", type=str, default="quants",
                    help="filename of quantification results")

######################### preprocessing #########################
parser.add_argument("--num_workers", type=int, default=4,
                    help="number of workers")
parser.add_argument("--min_prob", type=float, default=0.96,
                    help="probability threshold of detection model")

parser.add_argument("--roi_x", type=int, default=64,
                    help="roi size in X direction")
parser.add_argument("--roi_y", type=int, default=64,
                    help="roi size in Y direction")
parser.add_argument("--roi_z", type=int, default=64,
                    help="roi size in Z direction")
                    
parser.add_argument("--in_channels", type=int, default=1,
                    help="number of input channels")
parser.add_argument("--out_channels", type=int, default=2,
                    help="number of output channels")


args = parser.parse_args()
