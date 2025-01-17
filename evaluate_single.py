import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from gma.network import RAFTGMA
from gma.utils import flow_viz
from gma.utils.utils import InputPadder


# DEVICE = 'cuda'
DEVICE = "cpu"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (400, 300))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_dir):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    np.save(os.path.join(flow_dir, 'flo_numpy.npy'), flo, allow_pickle=True)

    # map flow to rgb image
    flo, _ = flow_viz.flow_to_image(flo, color_depth=16)
    print(flo.dtype)

    imageio.imwrite(os.path.join(flow_dir, 'flo.png'), flo, bits=16)
    cv2.imwrite(os.path.join(flow_dir, 'flo_opencv.png'), cv2.cvtColor(flo, cv2.COLOR_RGB2BGR))
    print(f"Saving optical flow visualisation at {os.path.join(flow_dir, 'flo.png')}")


def normalize(x):
    return x / (x.max() - x.min())


def demo(args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    flow_dir = os.path.join(args.path, args.model_name)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print(f"Reading in images at {imfile1} and {imfile2}")

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
            print(f"Estimating optical flow...")

            viz(image1, flow_up, flow_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
