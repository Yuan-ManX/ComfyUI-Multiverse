import argparse
from pathlib import Path
import os
import random
import h5py
import numpy as np
import cv2

low_res_w = 64
low_res_h = 48

crop_frame = {
    'left_top': (0.04, 0.18),
    'right_bottom': (0.92, 0.95)
}

def extract_roi(image, rect):
    img_width, img_height, _ = image.shape
    min_x = int(rect['left_top'][1] * img_width)
    max_x = int(rect['right_bottom'][1] * img_width)
    min_y = int(rect['left_top'][0] * img_height)
    max_y = int(rect['right_bottom'][0] * img_height)
    roi = image[min_x:max_x, min_y:max_y]
    return roi

def rescale_image(image, scale_factor):
    # Get new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "full_res_directory",
        type=Path,
        help="Specify your full_res directory.",
    )

    parser.add_argument(
        "model_directory",
        type=Path,
        help="Specify ",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    full_res_directory = args.full_res_directory.absolute()
    model_directory = args.model_directory.absolute()

    spawn_dir = model_directory / "game/spawn"
    existing_spawns = len(os.listdir(spawn_dir))
    os.makedirs(spawn_dir / str(existing_spawns), exist_ok=True)
    full_res = os.path.join(full_res_directory, random.choice(os.listdir(full_res_directory)))
    h5file = h5py.File(full_res, 'r')
    i = random.randint(0, 1000)

    data_x_frames = []
    data_y_frames = []
    next_act_frames = []
    low_res_frames = []

    for j in range(20):
        frame_x = f'frame_{i}_x'
        frame_y = f'frame_{i}_y'

        # Check if the datasets exist in the file
        if frame_x in h5file and frame_y in h5file:
            # Append each frame to the lists
            data_x = h5file[frame_x][:]
            data_y = h5file[frame_y][:]

            img1 = cv2.cvtColor(data_x[:,:,3:], cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(data_x[:,:,:3], cv2.COLOR_BGR2RGB)

            img1_cropped = extract_roi(img1, crop_frame)
            img2_cropped = extract_roi(img2, crop_frame)
            img1_cropped = cv2.resize(img1_cropped, (530, 350), interpolation=cv2.INTER_AREA)
            img2_cropped = cv2.resize(img2_cropped, (530, 350), interpolation=cv2.INTER_AREA)

            data_x_frames.append(np.concatenate([img1_cropped, img2_cropped], axis=2))
            data_y_frames.append(data_y)

            img1 = cv2.resize(img1, (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)

            low_res_frames.append(np.concatenate([img1, img2], axis=2).astype(np.uint8))
        else:
            print(f"One or both of {frame_x} or {frame_y} do not exist in the file.")
        i += 1
    for _ in range(200):
        next_act = f'frame_{i}_y'
        if next_act in h5file:
            next_act_data = h5file[next_act][:]
            next_act_frames.append(next_act_data)

    data_x_stacked = np.stack(data_x_frames)
    data_y_stacked = np.stack(data_y_frames)
    next_act_stacked = np.stack(next_act_frames)
    low_res_stacked = np.stack(low_res_frames)

    low_res_stacked = np.transpose(low_res_stacked, (0, 3, 1, 2))
    data_x_stacked = np.transpose(data_x_stacked, (0, 3, 1, 2))

    print(f"Saving act.npy of size {data_y_stacked.shape}")
    np.save(spawn_dir / f"{existing_spawns}/act.npy", data_y_stacked)
    print(f"Saving full_res.npy of size {data_x_stacked.shape}")
    np.save(spawn_dir / f"{existing_spawns}/full_res.npy", data_x_stacked)
    print(f"Saving next_act.npy of size {next_act_stacked.shape}")
    np.save(spawn_dir / f"{existing_spawns}/next_act.npy", next_act_stacked)
    print(f"Saving low_res.npy of size {low_res_stacked.shape}")
    np.save(spawn_dir / f"{existing_spawns}/low_res.npy", low_res_stacked)

    h5file.close()


if __name__ == "__main__":
    main()
