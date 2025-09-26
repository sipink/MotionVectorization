import kornia
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time


def downsample(img, scale):
    return F.avg_pool2d(img, scale, count_include_pad=False, stride=[1, 1])


def get_grid(w, h, bleed, device):
    grid_size = [w + 2 * bleed, h + 2 * bleed]
    xs, ys = np.meshgrid(
        np.linspace(-(w + 2 * bleed) / w, (w + 2 * bleed) / w, grid_size[0]),
        np.linspace(-(h + 2 * bleed) / h, (h + 2 * bleed) / h, grid_size[1]),
    )
    grid = np.stack([xs, ys]).reshape((2, -1)).T
    return torch.tensor(grid, dtype=torch.float64, device=device)


def torch2numpy(tensor):
    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    return tensor


def sampling_layer(
    img,
    x,
    y,
    scale_x,
    scale_y,
    theta,
    shear_x,
    shear_y,
    size,
    origin=None,
    blur=False,
    blur_kernel=3,
    bleed=0,
    interp="bilinear",
    device="cpu",
):
    """
    Renders a set to 2D canvas
    :param img: NxCxIHxIW input patch
    :param x: N center location x coordinate system on canvas is -1 to 1
    :param y: N center location y
    :param scale_x: N scale of the element wrt to canvas x ratio of length w.r.t to total canvas length
    :param scale_y: N scale of the element wrt to canvas y
    """
    t0 = time.perf_counter()
    b, c, h, w = img.shape
    repeat = x.shape[0]
    if origin is None:
        origin = torch.zeros((repeat, 2)).to(device)
    else:
        origin = 2 * origin - 1

    # NOTE: https://discuss.pytorch.org/t/rotating-non-square-images-using-affine-grid/21592/2.
    shear_matrix_x = torch.stack(
        [
            torch.ones_like(shear_x),
            -shear_x,
            torch.zeros_like(x),
            torch.zeros_like(shear_x),
            torch.ones_like(shear_x),
            torch.zeros_like(y),
        ],
        dim=1,
    ).view(-1, 2, 3)
    shear_matrix_y = torch.stack(
        [
            torch.ones_like(shear_y),
            torch.zeros_like(shear_y),
            torch.zeros_like(x),
            -shear_y,
            torch.ones_like(shear_y),
            torch.zeros_like(y),
        ],
        dim=1,
    ).view(-1, 2, 3)
    aff_matrix = torch.stack(
        [
            torch.cos(-theta),
            -torch.sin(-theta),
            -x,
            torch.sin(-theta),
            torch.cos(-theta),
            -y,
        ],
        dim=1,
    ).view(-1, 2, 3)
    aff_matrix = aff_matrix.to(device)
    A_batch = aff_matrix[:, :, :2]
    b_batch = aff_matrix[:, :, 2].unsqueeze(1)
    Kx_batch = shear_matrix_x[:, :, :2].to(device)
    Ky_batch = shear_matrix_y[:, :, :2].to(device)

    coords = get_grid(size[3], size[2], bleed, device).unsqueeze(0).repeat(repeat, 1, 1)
    sc = torch.stack([scale_x, scale_y], dim=-1).to(device)
    coords = coords - origin[:, None, :]
    coords = coords + b_batch
    coords = coords.bmm(A_batch.transpose(1, 2))
    coords = coords.bmm(Ky_batch.transpose(1, 2))
    coords = coords.bmm(Kx_batch.transpose(1, 2))
    coords = coords / sc[:, None, :]
    coords = coords + origin[:, None, :]

    grid = coords.view(-1, size[3] + 2 * bleed, size[2] + 2 * bleed, 2)
    if blur:
        img = kornia.filters.gaussian_blur2d(
            img, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel)
        )
    if b == 1:
        img = img.repeat(repeat, 1, 1, 1)
    render = F.grid_sample(img, grid, interp, "zeros")
    t1 = time.perf_counter()
    return render


if __name__ == "__main__":
    pass
