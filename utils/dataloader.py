import os
import numpy as np

from argparse import ArgumentParser as Parser
from torch.utils.data import DataLoader, Dataset

from utils import transforms as T


class MyDataset(Dataset):
    ''' Custom Dataset Class '''

    def __init__(self,
                 data_dir: str,
                 data_list: list,
                 transforms: object):
        ''' Args:
        * `data_dir`: save path of dataset.
        * `data_list`: filenames of data in this fold.
        * `transforms`: transform functions for dataset.
        '''
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, index: int):
        ''' get image, label and filename by index '''
        info = {"image": "", "label": None}
        for key, val in self.data_list[index].items():
            info[key] = os.path.join(self.data_dir, val)
        img, msk = self.transforms(info["image"], info["label"])
        info["fname"] = os.path.basename(info["image"])
        info["suid"] = info["fname"].split("_image")[0]
        info["suid"] = info["suid"].replace(".nii.gz", "")
        return img, msk, info

    def __len__(self):
        ''' get length of the dataset '''
        return len(self.data_list)


def get_loader(args: Parser):
    val_transforms = T.Compose([
        T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
        T.ScaleIntensity(scope=(-1200, 600), range=(-1, 1), clip=True),
        T.AddChannel(img_add=True, msk_add=True),
        T.ToTensor(device=args.device)
    ])

    data_list = get_data_list(args)
    test_dataset = MyDataset(data_dir=args.data_root,
                             data_list=data_list,
                             transforms=val_transforms)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)
    return test_loader


def get_data_list(args: Parser):
    data_list = []
    img_folder = os.path.join(args.data_root, args.image_dir)
    for fname in os.listdir(img_folder):
        img_path = os.path.join(args.image_dir, fname)
        msk_path = os.path.join(args.mask_dir, fname.replace("images", "masks"))
        data_list.append({"image": img_path, "label": msk_path})
    return data_list

