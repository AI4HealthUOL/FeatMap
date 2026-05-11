from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import cv2

"""Implementation of direct manipulations/transformations"""


def apply_rotation(img, degrees):
    return img.rotate(degrees)


def apply_translation(img, tx, ty):
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, tx, 0, 1, ty))


def apply_mirror(img, direction):
    if direction == "h":
        return ImageOps.mirror(img)
    elif direction == "v":
        return ImageOps.flip(img)
    return img


def apply_perspective_transform(img, warp_points):
    width, height = img.size
    src_points = np.float32(warp_points)
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_cv = np.array(img)
    warped = cv2.warpPerspective(img_cv, matrix, (width, height))
    return Image.fromarray(warped)


def apply_hue_shift(img, hue_shift):
    img_hsv = img.convert("HSV")
    np_img = np.array(img_hsv, dtype=np.int16)
    np_img[..., 0] = (np_img[..., 0] + hue_shift) % 256
    np_img = np_img.astype(np.uint8)
    img_hsv = Image.fromarray(np_img, mode="HSV")
    return img_hsv.convert("RGB")


def apply_grayscale(img):
    return img.convert("L").convert("RGB")


def apply_brightness(img, delta):
    enhancer = ImageEnhance.Brightness(img)
    factor = 1 + delta / 100.0
    return enhancer.enhance(factor)


def apply_contrast(img, delta):
    enhancer = ImageEnhance.Contrast(img)
    factor = 1 + delta / 100.0
    return enhancer.enhance(factor)


def apply_noise(img, amount):
    np_img = np.array(img)
    noise = np.random.normal(0, amount, np_img.shape).astype(np.int16)
    noisy_img = np.clip(np_img.astype(np.int16) +
                        noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def apply_mask(img, polygon_points, color):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(polygon_points, outline=255, fill=255)

    mask_np = np.array(mask)
    img_np = np.array(img)

    img_np[mask_np == 255] = color

    return Image.fromarray(img_np)
