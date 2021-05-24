import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import time

from gma.network import RAFTGMA
from gma.utils import flow_viz
from gma.utils.utils import InputPadder


def convert_frame(frame, device):
    img = torch.from_numpy(frame).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img, flo, flow_dir, frame_counter):
    # img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, f'flo_{frame_counter:03d}.png'), flo)
    # print(f"Saving optical flow visualisation at {os.path.join(flow_dir, 'flo.png')}")
    return flo


def demo(args):
    device = "cpu"
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(device)
    model.eval()

    flow_dir = os.path.join(args.path, args.model_name)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    video_reader = cv2.VideoCapture(args.video_path)
    ret, frame_previous = video_reader.read()
    frame_previous = convert_frame(frame_previous, device)
    padder = InputPadder(frame_previous.shape)
    # frame_previous = padder.pad(frame_previous)[0]
    frame_counter = 0
    processed_frame_counter = 0

    batch = []
    batch_frames = []

    video_writer = cv2.VideoWriter(
        "flow_out.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0 / args.subsampling_factor, (800, 600), True,
    )

    all_rad_max = []

    start = time.time()

    with torch.no_grad():
        while ret:
            while ret and len(batch) < args.batch_size + 1:
                ret, frame_current = video_reader.read()
                if not ret:
                    break

                if frame_counter % args.subsampling_factor == 0:
                    # print(f"Loading (flow) frame {frame_counter:03d}")
                    frame_current = convert_frame(frame_current, device)
                    frame_current = padder.pad(frame_current)[0]
                    batch.append(frame_current)
                    batch_frames.append(frame_counter)

                frame_counter += 1

            if len(batch) > 1:
                print(f"Processing frames {min(batch_frames):04d} - {max(batch_frames):04d} (#frames = {len(batch)}, "
                      f"{processed_frame_counter / (time.time() - start)} FPS)")

                batch_previous = torch.cat(batch[:-1])
                batch_current = torch.cat(batch[1:])
                flow_low, flow_up = model(batch_previous, batch_current, iters=12, test_mode=True)

                flo = flow_up.permute(0, 2, 3, 1).cpu().numpy()
                for f in flo:
                    f, rad_max = flow_viz.flow_to_image(f, clip_flow=args.flow_max)
                    video_writer.write(f)
                    all_rad_max.append(rad_max)

                processed_frame_counter += len(batch) - 1

                batch = batch[-1:]
                batch_frames = batch_frames[-1:]


            if not ret:
                break

    video_reader.release()
    video_writer.release()

    print(f"\nrad_max max: {np.max(all_rad_max)}")
    print(f"rad_max mean: {np.mean(all_rad_max)}")
    print(f"rad_max median: {np.median(all_rad_max)}")

    pcts = [99, 97, 95] + list(range(90, 40, -10))
    percentiles = np.percentile(all_rad_max, pcts)
    print(f"\nPercentiles: {pcts}")
    print(f"             {percentiles.tolist()}")

    plt.scatter(np.random.randn(len(all_rad_max)), all_rad_max)
    plt.savefig("new_imgs/rad_max_dist.png")
    plt.clf()

    plt.boxplot(all_rad_max)
    plt.savefig("new_imgs/rad_max_box.png")


if __name__ == '__main__':
    # TODO: properly specify whether GPU/CPU is supposed to be used, whether frames should be
    #  skipped/the video should be subsampled in time, and a batch size if the GPU should be used.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--video_path', help="video to computer flow for")
    parser.add_argument('--gpu', type=int, help="index of GPU to use (if not specified, CPU is used)")
    parser.add_argument('--flow_max', type=float, help="maximum flow to use for normalization")
    parser.add_argument('--subsampling_factor', type=int, default=1,
                        help="factor by which to subsample video in time dimension")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size for GPU inference")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
