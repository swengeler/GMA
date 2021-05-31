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


def direct_encoding(flo, max_value=None, preserve_direction=False, **kwargs):
    if max_value is None:
        max_value = np.max(flo)

    if not preserve_direction:
        flo = np.clip(flo, -max_value, max_value)
        flo = flo / (max_value * 2)
    else:
        pass

    frame = (flo * 255.0 + 127.0).astype(np.uint8)
    frame = np.concatenate((frame, np.zeros(frame.shape[:2] + (1,), dtype=np.uint8)), axis=2)
    return frame


def opencv_encoding(flo, max_value=None, **kwargs):
    if max_value is None:
        max_value = np.max(flo)

    mag, ang = cv2.cartToPolar(flo[:, :, 0], flo[:, :, 1])

    frame = np.zeros(flo.shape[:2] + (3,), dtype=np.uint8)
    frame[:, :, 2] = 255
    frame[:, :, 0] = (180.0 * ang / (2 * np.pi)).astype(np.uint8)
    frame[:, :, 1] = (255.0 * np.clip(mag, 0, max_value) / max_value).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)

    return frame


def demo(args):
    device = "cpu"
    if args.gpu is not None:
        device = f"cuda:{args.gpu[0]}"

    model = torch.nn.DataParallel(RAFTGMA(args), device_ids=None if args.gpu is None else args.gpu)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded checkpoint at {args.model}")

    # model = model.module
    model.to(device)
    # model.cuda()
    model.eval()

    flow_dir = os.path.join(args.path, args.model_name)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    all_flow_below_one = []

    video_reader = cv2.VideoCapture(args.video_path)
    w, h, fps = (video_reader.get(i) for i in range(3, 6))
    w, h = int(w), int(h)
    padder = InputPadder((h, w, 3))

    frame_counter = 0
    processed_frame_counter = 0

    batch = []
    batch_frames = []

    video_writer = cv2.VideoWriter(
        os.path.join(args.path, f"{args.out_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.subsampling_factor, (w, h), True,
    )

    encoding_func = {
        "direct": direct_encoding,
        "gma": lambda x, *regargs, **kwargs: flow_viz.flow_to_image(
            x, *regargs, **kwargs, clip_magnitude=kwargs.get("max_value", None))[0],
        "opencv": opencv_encoding,
    }[args.encoding]

    all_rad_max = []

    start = time.time()
    data_loading_time = 0
    data_saving_time = 0
    inference_time = 0

    with torch.no_grad():
        while ret:
            while ret and len(batch) < args.batch_size + 1:
                ret, frame_current = video_reader.read()
                if not ret:
                    break

                if frame_counter % args.subsampling_factor == 0:
                    start_data_loading_time = time.time()
                    # print(f"Loading (flow) frame {frame_counter:03d}")
                    frame_current = convert_frame(cv2.cvtColor(frame_current, cv2.COLOR_BGR2RGB), device)
                    frame_current = padder.pad(frame_current)[0]
                    batch.append(frame_current)
                    batch_frames.append(frame_counter)
                    data_loading_time += time.time() - start_data_loading_time

                frame_counter += 1

            if len(batch) > 1:
                start_inference_time = time.time()
                batch_previous = torch.cat(batch[:-1])
                batch_current = torch.cat(batch[1:])
                flow_low, flow_up = model(batch_previous, batch_current, iters=12, test_mode=True)
                inference_time += time.time() - start_inference_time

                start_data_saving_time = time.time()
                flo = flow_up.permute(0, 2, 3, 1).cpu().numpy()
                for f_idx, f in enumerate(flo):
                    flow_below_one = np.sum(np.linalg.norm(np.reshape(f, (-1, 2)), axis=1) < 1)
                    all_flow_below_one.append(flow_below_one)
                    # f, rad_max = flow_viz.flow_to_image(f, clip_magnitude=args.flow_max)
                    # f = direct_encoding(f, max_value=args.flow_max)
                    f = cv2.cvtColor(encoding_func(f, max_value=args.flow_max), cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(os.path.join(args.path, args.model_name, f"{processed_frame_counter+f_idx:04d}.png"), f)
                    video_writer.write(f)
                    # all_rad_max.append(rad_max)
                data_saving_time += time.time() - start_data_saving_time

                processed_frame_counter += len(batch) - 1

                time_total = time.time() - start
                """
                print(f"Processed frames {min(batch_frames):04d} - {max(batch_frames):04d} (#frames = {len(batch)}, "
                      f"{processed_frame_counter / time_total:02.4f} FPS) - Time spent on loading, inference, saving: "
                      f"{data_loading_time / time_total * 100:02.2f}%, {inference_time / time_total * 100:02.2f}%, "
                      f"{data_saving_time / time_total * 100:02.2f}%")
                """

                batch = batch[-1:]
                batch_frames = batch_frames[-1:]

            if not ret:
                break

    video_reader.release()
    video_writer.release()

    if len(all_rad_max) > 0:
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

    if len(all_flow_below_one) > 0:
        all_flow_below_one = np.array(all_flow_below_one)
        np.save(f"new_imgs/{args.out_name}_fbop.npy", all_flow_below_one)
        plt.plot(np.arange(len(all_flow_below_one))[~np.isnan(all_flow_below_one)],
                 all_flow_below_one[~np.isnan(all_flow_below_one)])
        plt.savefig(f"new_imgs/{args.out_name}_fbop.png")
        plt.clf()


if __name__ == '__main__':
    # TODO: properly specify whether GPU/CPU is supposed to be used, whether frames should be
    #  skipped/the video should be subsampled in time, and a batch size if the GPU should be used.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--out_name', help="name of the output video", default="flow")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--video_path', help="video to computer flow for")
    parser.add_argument('--gpu', type=int, nargs="+", help="index of GPU to use (if not specified, CPU is used)")
    parser.add_argument('--flow_max', type=float, help="maximum flow to use for normalization")
    parser.add_argument('--subsampling_factor', type=int, default=1,
                        help="factor by which to subsample video in time dimension")
    parser.add_argument('--encoding', default="direct", choices=["direct", "gma", "opencv"],
                        help="encoding for the flow output")
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

