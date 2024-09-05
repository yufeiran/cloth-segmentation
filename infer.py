import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms




from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda"

#image_dir = "input_images"
#result_dir = "output_images"
checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"trained_checkpoint", "cloth_segm_u2net_latest.pth")
do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

class ClothSegemntation():

    def __init__(self, image_dir, result_dir):
        self.image_dir = image_dir
        self.result_dir = result_dir
        self.checkpoint_path = checkpoint_path
        self.do_palette = do_palette

        if(not os.path.exists(self.result_dir)):
            os.makedirs(self.result_dir)


    def infer(self):
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)

        net = U2NET(in_ch=3, out_ch=4)
        net = load_checkpoint_mgpu(net, checkpoint_path)
        net = net.to(device)
        net = net.eval()

        palette = get_palette(4)

        images_list = sorted(os.listdir(self.image_dir))
        pbar = tqdm(total=len(images_list))
        for image_name in images_list:
            img = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
            image_tensor = transform_rgb(img)
            image_tensor = torch.unsqueeze(image_tensor, 0)

            output_tensor = net(image_tensor.to(device))
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy()

            output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
            if do_palette:
                output_img.putpalette(palette)
            output_img.save(os.path.join(self.result_dir, image_name[:-3] + "png"))

            # using raw img and output img to create segmented image
            raw_img = np.array(img)
            mask_img = np.array(output_img)
            upperbody_img = np.zeros_like(raw_img)
            lowerbody_img = np.zeros_like(raw_img)
            fullbody_img = np.zeros_like(raw_img)

            # if mask_img color in rgb is red, then it is upperbody
            upperbody_img[mask_img == 1] = raw_img[mask_img == 1]


            # if mask_img color in rgb is green, then it is lowerbody
            lowerbody_img[mask_img == 2] = raw_img[mask_img == 2]

            # if mask_img color in rgb is yellow, then it is fullbody
            fullbody_img[mask_img == 3] = raw_img[mask_img == 3]


            Image.fromarray(upperbody_img).save(os.path.join(self.result_dir, image_name[:-4] + "_upperbody.png"))
            Image.fromarray(lowerbody_img).save(os.path.join(self.result_dir, image_name[:-4] + "_lowerbody.png"))
            # Image.fromarray(fullbody_img).save(os.path.join(result_dir, image_name[:-4] + "_fullbody.png"))




            pbar.update(1)

        pbar.close()