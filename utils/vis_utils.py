"""
Save a couple images to grids with cond, render cond, novel render, novel gt
Also save images to a render video
"""
import glob
import os
from PIL import Image
import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from utils.sh_utils import eval_sh, SH2RGB
from einops import rearrange

im_res = 128

def gridify():

    out_folder = "grids_objaverse"
    os.makedirs(out_folder, exist_ok=True)

    folder_paths = glob.glob("/scratch/shared/beegfs/stan/scaling_splatter_image/objaverse/*")
    # pixelnerf_root = "/scratch/shared/beegfs/stan/splatter_image/pixelnerf/teddybears"
    folder_paths_test = sorted([fpath for fpath in folder_paths if "gt" not in fpath], key= lambda x: int(os.path.basename(x).split("_")[0]))
    """folder_paths_test = [folder_paths_test[i] for i in [5, 7, 12, 15,
                                                        18, 19, 30, 33,
                                                        37, 42, 43, 44,
                                                        48, 51, 64, 66,
                                                        70, 74, 78, 85, 
                                                        89, 91, 92]]"""

    # Initialize variables for grid dimensions
    num_examples_row = 6
    rows = num_examples_row
    num_per_ex = 2
    cols = num_examples_row * num_per_ex # 7 * 2
    im_res = 128

    for im_idx in range(100):
        print("Doing frame {}".format(im_idx))
        # for im_name in ["xyz", "colours", "opacity", "scaling"]:
        grid = np.zeros((rows*im_res, cols*im_res, 3), dtype=np.uint8)

        # Iterate through the folders in the out_folder
        for f_idx, folder_path_test in enumerate(folder_paths_test[:num_examples_row*num_examples_row]):
            # if im_name == "xyz":
            #     print(folder_path_test)
            row_idx = f_idx // num_examples_row
            col_idx = f_idx % num_examples_row
            im_path = os.path.join(folder_path_test, "{:05d}.png".format(im_idx))
            im_path_gt = os.path.join(folder_path_test + "_gt", "{:05d}.png".format(im_idx))
            """im_path_pixelnerf = os.path.join(pixelnerf_root, os.path.basename(folder_path_test),
                                             "{:06d}.png".format(im_idx))"""

            # im_path = os.path.join(folder_path_test, "{}.png".format(im_name))
            try:
                im = np.array(Image.open(im_path))
                im_gt = np.array(Image.open(im_path_gt))
                #im_pn = np.array(Image.open(im_path_pixelnerf))
                grid[row_idx*im_res: (row_idx+1)*im_res,
                 col_idx * num_per_ex *im_res: (col_idx * num_per_ex+1)*im_res, : ] = im[:, :, :3]
                grid[row_idx*im_res: (row_idx+1)*im_res,
                 (col_idx * num_per_ex + 1) *im_res: (col_idx* num_per_ex +2)*im_res, : ] = im_gt[:, :, :3]
                """grid[row_idx*im_res: (row_idx+1)*im_res,
                 (col_idx * num_per_ex + 2) *im_res: (col_idx* num_per_ex +3)*im_res, : ] = im_pn[:, :, :3]"""
            except FileNotFoundError:
                a = 0
        im_out = Image.fromarray(grid)
        im_out.save(os.path.join(out_folder, "{:05d}.png".format(im_idx)))
        # im_out.save(os.path.join(out_folder, "{}.png".format(im_name)))

def comparisons():

    out_root = "hydrants_comparisons"
    os.makedirs(out_root, exist_ok=True)

    folder_paths = glob.glob("/users/stan/pixel-nerf/full_eval_hydrant/*")
    folder_paths_test = sorted(folder_paths)
    folder_paths_ours_root = "/scratch/shared/beegfs/stan/out_hydrants_with_lpips_ours"

    # Initialize variables for grid dimensions
    rows = 3
    cols = 1
    im_res = 128

    for f_idx, folder_path_test in enumerate(folder_paths_test):

        example_id = "_".join(os.path.basename(folder_path_test).split("_")[1:])
        out_folder = os.path.join(out_root, example_id)
        os.makedirs(out_folder, exist_ok=True)
        num_images = len([p for p in glob.glob(os.path.join(folder_path_test, "*.png")) if "gt" not in p])

        grid = np.zeros((rows*im_res, cols*im_res, 3), dtype=np.uint8)

        for im_idx in range(num_images):

            im_path_pixelnerf = os.path.join(folder_path_test, "{:06d}.png".format(im_idx+1))
            im_path_ours = os.path.join(folder_paths_ours_root, example_id, "{:05d}.png".format(im_idx))
            im_path_gt = os.path.join(folder_paths_ours_root, example_id + "_gt", "{:05d}.png".format(im_idx))
            # im_path = os.path.join(folder_path_test, "{}.png".format(im_name))

            im_pn = np.array(Image.open(im_path_pixelnerf))
            im_ours = np.array(Image.open(im_path_ours))
            im_gt = np.array(Image.open(im_path_gt))

            grid[:im_res, :, :] = im_pn
            grid[im_res:2*im_res, :, :] = im_ours
            grid[2*im_res:3*im_res, :, :] = im_gt

            im_out = Image.fromarray(grid)
            im_out.save(os.path.join(out_folder, "{:05d}.png".format(im_idx)))

def vis_image_preds(recon: dict, folder_out: str):
    """
    Visualises network's image predictions.
    Args:
        recon: a dictionary of xyz, opacity, scaling, rotation, features_dc and features_rest
    """

    os.makedirs(folder_out, exist_ok=True)

    for i, image_preds in enumerate(recon):

        image_preds_reshaped = {}
        ray_dirs = (image_preds["xyz"].detach().cpu() / torch.norm(image_preds["xyz"].detach().cpu(), dim=-1, keepdim=True)).reshape(im_res, im_res, 3)

        for k, v in image_preds.items():
            image_preds_reshaped[k] = v
            if k == "xyz":
                # image_preds_reshaped[k] = (image_preds_reshaped[k] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]) / (
                #     torch.max(image_preds_reshaped[k], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]
                # )
                image_preds_reshaped[k] = normalize_tensor(image_preds_reshaped[k])
            if k == "scaling":
                # image_preds_reshaped["scaling"] = (image_preds_reshaped["scaling"] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]) / (
                #     torch.max(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]
                # )
                image_preds_reshaped[k] = normalize_tensor(image_preds_reshaped[k])
            if k != "features_rest":
                image_preds_reshaped[k] = image_preds_reshaped[k].reshape(im_res, im_res, -1).detach().cpu()
            else:
                image_preds_reshaped[k] = image_preds_reshaped[k].reshape(im_res, im_res, 3, 3).detach().cpu().permute(0, 1, 3, 2)
            if k == "opacity":
                image_preds_reshaped[k] = image_preds_reshaped[k].expand(im_res, im_res, 3) 


        colours = torch.cat([image_preds_reshaped["features_dc"].unsqueeze(-1), image_preds_reshaped["features_rest"]], dim=-1)
        colours = eval_sh(1, colours, ray_dirs)

        opacity = normalize_tensor(image_preds_reshaped["opacity"])
        colours = normalize_tensor(colours * opacity + 1 - opacity)
        xyz = normalize_tensor(image_preds_reshaped["xyz"] * opacity + 1 - opacity)
        scaling = normalize_tensor(image_preds_reshaped["scaling"] * opacity + 1 - opacity)

        plt.imsave(os.path.join(folder_out, f"colours_{i+1}.png"),
                colours.numpy())
        plt.imsave(os.path.join(folder_out, f"opacity_{i+1}.png"),
                opacity.numpy())
        plt.imsave(os.path.join(folder_out, f"xyz_{i+1}.png"), 
                xyz.numpy())
        plt.imsave(os.path.join(folder_out, f"scaling_{i+1}.png"), 
                scaling.numpy())

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

# Function to create rotation matrices
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
                    [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                    [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]])



def vis_gaussian_pos(image_preds: dict, make_video: bool, folder_out: str, splat: str):
    """
    Plot Gaussian positions in 3D space and save images from different camera angles.
    
    Args:
        recon (dict): Dictionary containing Reconstruction
        camera_poses (list): List of camera poses (4x4 transformation matrices).
        folder_out (str): Output folder to save the rendered images.
    """
    # Extract 3D positions from the recon dictionary
    positions = image_preds['xyz'].squeeze(0).detach().cpu().numpy()

    # Extract colors
    ray_dirs = (image_preds["xyz"].detach().cpu() / torch.norm(image_preds["xyz"].detach().cpu(), dim=-1, keepdim=True)).reshape(im_res, im_res, 3)
    image_preds_reshaped = {}
    for k, v in image_preds.items():
        image_preds_reshaped[k] = v
        if k == "xyz":
            # image_preds_reshaped[k] = (image_preds_reshaped[k] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]) / (
            #     torch.max(image_preds_reshaped[k], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]
            # )
            image_preds_reshaped[k] = normalize_tensor(image_preds_reshaped[k])
        if k == "scaling":
            # image_preds_reshaped["scaling"] = (image_preds_reshaped["scaling"] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]) / (
            #     torch.max(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]
            # )
            image_preds_reshaped[k] = normalize_tensor(image_preds_reshaped[k])
        if k != "features_rest":
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(im_res, im_res, -1).detach().cpu()
        else:
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(im_res, im_res, 3, 3).detach().cpu().permute(0, 1, 3, 2)
        if k == "opacity":
            image_preds_reshaped[k] = image_preds_reshaped[k].expand(im_res, im_res, 3) 


    colours = torch.cat([image_preds_reshaped["features_dc"].unsqueeze(-1), image_preds_reshaped["features_rest"]], dim=-1)
    colours = eval_sh(1, colours, ray_dirs)

    opacity = normalize_tensor(image_preds_reshaped["opacity"])
    colours = normalize_tensor(colours * opacity + 1 - opacity).reshape(-1, 3)

    # Setup plot
    opacity = image_preds['opacity'].squeeze(0).detach().cpu().numpy()
    threshold = 0.05 if splat == 'front' else 0.03
    mask = (opacity > threshold).all(axis=-1)  # Only show where opacity > threshold

    # Show only good position
    positions = image_preds['xyz'].squeeze(0).detach().cpu().numpy()[mask]
    colours = colours[mask]

    # Compute the centroid of the object
    centroid = positions.mean(axis=0)

    # Center the positions at the origin (object space)
    centered_positions = positions - centroid

    # Create a new 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    ax.view_init(elev=-90, azim=-90)

    # Plot the Gaussian positions
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                         c=colours, marker='o', s=10)

    # Set equal aspect ratio
    fixed_range = 0.35

    mid_x = 0  # Center at 0
    mid_y = 0.05  # Center at 0
    mid_z = 0  # Center at 0

    ax.set_xlim(mid_x - fixed_range, mid_x + fixed_range)
    ax.set_ylim(mid_y - fixed_range, mid_y + fixed_range)
    ax.set_zlim(mid_z - fixed_range, mid_z + fixed_range)

    if not make_video:
        # Create sliders for controlling rotation around x, y, z axes
        ax_rot_x = plt.axes([0.25, 0.25, 0.65, 0.03])
        ax_rot_y = plt.axes([0.25, 0.2, 0.65, 0.03])
        ax_rot_z = plt.axes([0.25, 0.15, 0.65, 0.03])

        slider_rot_x = Slider(ax_rot_x, 'Rot X', -180, 180, valinit=0)
        slider_rot_y = Slider(ax_rot_y, 'Rot Y', -180, 180, valinit=0)
        slider_rot_z = Slider(ax_rot_z, 'Rot Z', -180, 180, valinit=0)

        # Function to update the plot's data based on slider values
        def update(val):
            rot_x = np.radians(slider_rot_x.val)
            rot_y = np.radians(slider_rot_y.val)
            rot_z = np.radians(slider_rot_z.val)

            # Create rotation matrices
            rx = rotation_matrix([1, 0, 0], rot_x)
            ry = rotation_matrix([0, 1, 0], rot_y)
            rz = rotation_matrix([0, 0, 1], rot_z)

            # Combine rotations in object space
            rotation = rx @ ry @ rz

            # Apply rotation to centered positions and update scatter plots
            rotated_positions = centered_positions[i] @ rotation.T
            rotated_positions += centroid  # Translate back to the original centroid
            scatter._offsets3d = (rotated_positions[:, 0],
                                rotated_positions[:, 1],
                                rotated_positions[:, 2])

            # Redraw the figure
            fig.canvas.draw_idle()

        # Attach the update function to slider events
        slider_rot_x.on_changed(update)
        slider_rot_y.on_changed(update)
        slider_rot_z.on_changed(update)

        plt.show()

    # Save rotating frames if specified
    else:
        num_frames = 360

        os.makedirs(folder_out, exist_ok=True)  # Create output folder if it doesn't exist
        angles = np.linspace(0, 2 * np.pi, num_frames)  # Rotate from 0 to 360 degrees (in radians)

        for i, angle in enumerate(angles):
            ry = rotation_matrix([0, 1, 0], angle)  # Rotation matrix around Y-axis
            
            rotated_positions = centered_positions @ ry.T  # Rotate around Y-axis
            rotated_positions += centroid  # Translate back to the original centroid
            scatter._offsets3d = (rotated_positions[:, 0],
                                    rotated_positions[:, 1],
                                    rotated_positions[:, 2])

            # Save each frame as a PNG
            frame_filename = os.path.join(folder_out, f"{splat}gaussians_{i:03d}.png")
            fig.savefig(frame_filename)
            print(f"Saved frame {i+1}/{num_frames} to {frame_filename}")

    plt.close(fig)

def vis_gaussian_pos_two(two_splatter: tuple, make_video: bool, folder_out: str):
    """
    Plot Gaussian positions in 3D space and save images from different camera angles.
    
    Args:
        two_splatter (tuple): Tuple containing two sets of Gaussian positions and properties.
        camera_poses (list): List of camera poses (4x4 transformation matrices).
        folder_out (str): Output folder to save the rendered images.
    """
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    thresholds = [0.05, 0.03]
    scatter_plots = []  # To store scatter plots for each splatter
    pos = []  # To store all positions from both splatters for centroid calculation

    for i, image_preds in enumerate(two_splatter):
        # Opacity mask
        opacity = image_preds['opacity'].squeeze(0).detach().cpu().numpy()
        print(f'Image {i+1} Opacity: Max= {max(opacity)}, Min= {min(opacity)}')
        mask = (opacity > thresholds[i]).all(axis=-1)

        # Get positions
        positions = image_preds['xyz'].squeeze(0).detach().cpu().numpy()[mask]
        pos.append(positions)

        # Plot each set of Gaussian positions with a unique color
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', s=10)
        scatter_plots.append(scatter)

    # Combine all positions into one array for operations like rotation
    pos_combined = np.concatenate(pos, axis=0)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gaussian Positions')

    # Compute the centroid of the combined positions
    centroid = pos_combined.mean(axis=0)

    # Center the positions at the origin (object space)
    centered_positions = [p - centroid for p in pos]

    ax.view_init(elev=-90, azim=-90)
    
    # Set equal aspect ratio
    fixed_range = 0.35

    mid_x = 0  # Center at 0
    mid_y = 0.05  # Center at 0
    mid_z = 0  # Center at 0

    ax.set_xlim(mid_x - fixed_range, mid_x + fixed_range)
    ax.set_ylim(mid_y - fixed_range, mid_y + fixed_range)
    ax.set_zlim(mid_z - fixed_range, mid_z + fixed_range)

    if not make_video:
        # Create sliders for controlling rotation around x, y, z axes
        ax_rot_x = plt.axes([0.25, 0.25, 0.65, 0.03])
        ax_rot_y = plt.axes([0.25, 0.2, 0.65, 0.03])
        ax_rot_z = plt.axes([0.25, 0.15, 0.65, 0.03])

        slider_rot_x = Slider(ax_rot_x, 'Rot X', -180, 180, valinit=0)
        slider_rot_y = Slider(ax_rot_y, 'Rot Y', -180, 180, valinit=0)
        slider_rot_z = Slider(ax_rot_z, 'Rot Z', -180, 180, valinit=0)

        # Function to update the plot's data based on slider values
        def update(val):
            rot_x = np.radians(slider_rot_x.val)
            rot_y = np.radians(slider_rot_y.val)
            rot_z = np.radians(slider_rot_z.val)

            # Create rotation matrices
            rx = rotation_matrix([1, 0, 0], rot_x)
            ry = rotation_matrix([0, 1, 0], rot_y)
            rz = rotation_matrix([0, 0, 1], rot_z)

            # Combine rotations in object space
            rotation = rx @ ry @ rz

            # Apply rotation to centered positions and update scatter plots
            for i, scatter in enumerate(scatter_plots):
                rotated_positions = centered_positions[i] @ rotation.T
                rotated_positions += centroid  # Translate back to the original centroid
                scatter._offsets3d = (rotated_positions[:, 0],
                                    rotated_positions[:, 1],
                                    rotated_positions[:, 2])

            # Redraw the figure
            fig.canvas.draw_idle()

        # Attach the update function to slider events
        slider_rot_x.on_changed(update)
        slider_rot_y.on_changed(update)
        slider_rot_z.on_changed(update)

        plt.show()

    # Save rotating frames if specified
    else:
        num_frames = 360

        os.makedirs(folder_out, exist_ok=True)  # Create output folder if it doesn't exist
        angles = np.linspace(0, 2 * np.pi, num_frames)  # Rotate from 0 to 360 degrees (in radians)

        for i, angle in enumerate(angles):
            ry = rotation_matrix([0, 1, 0], angle)  # Rotation matrix around Y-axis
            
            for j, scatter in enumerate(scatter_plots):
                rotated_positions = centered_positions[j] @ ry.T  # Rotate around Y-axis
                rotated_positions += centroid  # Translate back to the original centroid
                scatter._offsets3d = (rotated_positions[:, 0],
                                      rotated_positions[:, 1],
                                      rotated_positions[:, 2])

            # Save each frame as a PNG
            frame_filename = os.path.join(folder_out, f"frame_{i:03d}.png")
            fig.savefig(frame_filename)
            print(f"Saved frame {i+1}/{num_frames} to {frame_filename}")

    plt.close(fig)


if __name__ == "__main__":
    gridify()