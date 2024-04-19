import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image, ImageEnhance
from PIL import Image
from typing import Literal, Any
from torch import Tensor, dtype
import torchvision.transforms.functional as D
import math

def radialspace_2D(
            size: tuple[int] | tuple[int, int],
            curvy: float = 1.0,
            scale: float = 1.0,
            mode: str = "square",
            min_val: float = 0.0,
            max_val: float = 1.0,
            center: tuple[float, float] = (0.5, 0.5),
            function: Any = None,
            normalize: bool = True,
            dtype: dtype = torch.float32
    ) -> Tensor:
    
        if isinstance(size, tuple):
            if len(size) == 1:
                width = height = size[0]
            elif len(size) == 2:
                width, height = size
            else:
                raise ValueError("Invalid size argument")
        else:
            raise TypeError("Size must be a tuple")

        x = torch.linspace(0, 1, width)
        y = torch.linspace(0, 1, height)

        xx, yy = torch.meshgrid(x, y, indexing='ij')

        if function is not None:
            d = function(xx, yy)
        elif mode == "square":
            xx = (torch.abs(xx - center[0]) ** curvy)
            yy = (torch.abs(yy - center[1]) ** curvy)
            d = torch.max(xx, yy)
        elif mode == "circle":
            d = torch.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
            d = (d ** curvy)
        elif mode == "rectangle":
            xx = (torch.abs(xx - center[0]) ** curvy)
            yy = (torch.abs(yy - center[1]) ** curvy)
            d = xx + yy
        elif mode == "corners":
            xx = (torch.abs(xx - center[0]) ** curvy)
            yy = (torch.abs(yy - center[1]) ** curvy)
            d = torch.min(xx, yy)
        else:
            raise ValueError("Not supported mode.")

        if normalize:
            d = d / d.max() * scale
            d = torch.clamp(d, min_val, max_val)

        return d.to(dtype)

def cv2_layer(tensor: Tensor, function) -> Tensor:
    """
    This function applies a given function to each channel of an input tensor and returns the result as a PyTorch tensor.

    :param tensor: A PyTorch tensor of shape (H, W, C) or (N, H, W, C), where C is the number of channels, H is the height, and W is the width of the image.
    :param function: A function that takes a numpy array of shape (H, W, C) as input and returns a numpy array of the same shape.
    :return: A PyTorch tensor of the same shape as the input tensor, where the given function has been applied to each channel of each image in the tensor.
    """
    shape_size = tensor.shape.__len__()

    def produce(image):
        channels = image[0, 0, :].shape[0]

        rgb = image[:, :, 0:3].numpy()
        result_rgb = function(rgb)

        if channels <= 3:
            return torch.from_numpy(result_rgb)
        elif channels == 4:
            alpha = image[:, :, 3:4].numpy()
            result_alpha = function(alpha)[..., np.newaxis]
            result_rgba = np.concatenate((result_rgb, result_alpha), axis=2)

            return torch.from_numpy(result_rgba)

    if shape_size == 3:
        return torch.from_numpy(produce(tensor))
    elif shape_size == 4:
        return torch.stack([
            produce(tensor[i]) for i in range(len(tensor))
        ])
    else:
        raise ValueError("Incompatible tensor dimension.")

def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = []
        for img in image:
            images.append(func(self, img, *args, **kwargs))
        batch_tensor = torch.cat(images, dim=0)
        return (batch_tensor, )
    return wrapper

def gamma_correction_pil(image, gamma):
    # Convert PIL Image to NumPy array
    img_array = np.array(image)
    # Normalization [0,255] -> [0,1]
    img_array = img_array / 255.0
    # Apply gamma correction
    img_corrected = np.power(img_array, gamma)
    # Convert corrected image back to original scale [0,1] -> [0,255]
    img_corrected = np.uint8(img_corrected * 255)
    # Convert NumPy array back to PIL Image
    corrected_image = Image.fromarray(img_corrected)
    return corrected_image


def apply_hald_clut(lut_image, img):
    hald_w, hald_h = lut_image.size
    clut_size = int(round(pow(hald_w, 1/3)))
    scale = (clut_size * clut_size - 1) / 255
    img = np.asarray(img)

    # Convert the HaldCLUT image to numpy array
    hald_img_array = np.asarray(lut_image)

    # If the HaldCLUT image is monochrome, duplicate its single channel to three
    if len(hald_img_array.shape) == 2:
        hald_img_array = np.stack([hald_img_array]*3, axis=-1)

    hald_img_array = hald_img_array.reshape(clut_size ** 6, 3)

    clut_r = np.rint(img[:, :, 0] * scale).astype(int)
    clut_g = np.rint(img[:, :, 1] * scale).astype(int)
    clut_b = np.rint(img[:, :, 2] * scale).astype(int)
    filtered_image = np.zeros((img.shape))
    filtered_image[:, :] = hald_img_array[clut_r + clut_size ** 2 * clut_g + clut_size ** 4 * clut_b]
    filtered_image = Image.fromarray(filtered_image.astype('uint8'), 'RGB')
    return filtered_image


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2numpy(image):
    # Convert tensor to numpy array and transpose dimensions from (C, H, W) to (H, W, C)
    return (255.0 * image.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)



class Blend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": ( [
                        "normal",
                        "multiply",
                        "screen",
                        "overlay",
                        "soft_light",
                        "hard_light",
                        "darken",
                        "lighten",
                        "color_dodge",
                        "color_burn",
                        "linear_dodge",
                        "linear_burn",
                        "linear_light",
                        "vivid_light",
                        "pin_light",
                        "difference",
                        "exclusion",
                        "divide"
                    ],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "postprocessing/Blends"

    def blend_images(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        blend_factor: float,
        blend_mode: str,
    ):
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "divide":
            return img1 / img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2-1)*(self.g(img1)-img1))
        elif mode == 'hard_light':
            return torch.where(img2 < 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == 'darken':
            return torch.min(img1, img2)
        elif mode == 'lighten':
            return torch.max(img1, img2)
        elif mode == 'color_dodge':
            return torch.where((img1 < 1), img1 / (1 - img2), torch.ones_like(img1))
        elif mode == 'color_burn':
            return torch.where(img2 > 0, (1 - ((1 - img1) / img2)), torch.zeros_like(img1))
        elif mode == 'linear_dodge':
            return img1 + img2
        elif mode == 'linear_burn':
            return img1 + img2 - 1
        elif mode == 'linear_light':
            return torch.where(img2 <= 0.5, img1 + 2 * (img2 - 0.5), img1 + 2 * img2 - 1)
        elif mode == 'vivid_light':
            return torch.where(img2 <= 0.5, (1 -(1-img1) / (2*img2)),img1/(2*(1-img2)))
        elif mode == 'pin_light':
            return torch.where(img2 < 0.5, torch.min(img1, 2*img2), torch.max(img1, 2*(img2-0.5)))
        elif mode == 'difference':
            return torch.abs(img1 - img2)
        elif mode == 'exclusion':
            return img1 + img2 - 2 * img1 * img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

    def crop_and_resize(self, img: torch.Tensor, target_shape: tuple):
        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img


        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img


class AlphaBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image2_mask":("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": ( [
                        "normal",
                        "add",
                    ],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "postprocessing/Blends"

    def blend_images(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        image2_mask: torch.Tensor,
        blend_factor: float,
        blend_mode: str,
    ):
        
        image1 = image1.to('cuda:0')
        image2 = image2.to('cuda:0')
        image2_mask = image2_mask.to('cuda:0')

        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)
            
        
        if image1.shape != image2_mask.shape:
            image2_mask =  self.crop_and_resize(image2_mask, image1.shape)

        blended_image = self.blend_mode(image1, image2, image2_mask, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2,img2_mask, mode):
        if mode == "normal":
            return img1*img2_mask+img2*(1-img2_mask)
        elif mode == "add":
            return img1+img2*(1-img2_mask)
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

    def crop_and_resize(self, img: torch.Tensor, target_shape: tuple):
        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img


        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img

class Blur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "postprocessing/Filters"

    def blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred.permute(0, 2, 3, 1)

        return (blurred,)

class CannyEdgeMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lower_threshold": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "upper_threshold": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canny"

    CATEGORY = "postprocessing/Masks"

    def canny(self, image: torch.Tensor, lower_threshold: float, upper_threshold: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)
        lower_threshold*=500
        upper_threshold*=500

        for b in range(batch_size):
            tensor_image = image[b].numpy().copy()
            gray_image = (cv2.cvtColor(tensor_image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            canny = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            tensor = torch.from_numpy(canny)
            result[b] = tensor

        return (result,)

class ChromaticAberration:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "red_direction": (["horizontal", "vertical"],),
                "green_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "green_direction": (["horizontal", "vertical"],),
                "blue_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "blue_direction": (["horizontal", "vertical"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "chromatic_aberration"

    CATEGORY = "postprocessing/Effects"

    def chromatic_aberration(self, image: torch.Tensor, red_shift: int, green_shift: int, blue_shift: int, red_direction: str, green_direction: str, blue_direction: str):
        def get_shift(direction, shift):
            shift = -shift if direction == 'vertical' else shift # invert vertical shift as otherwise positive actually shifts down
            return (shift, 0) if direction == 'vertical' else (0, shift)

        x = image.permute(0, 3, 1, 2)
        shifts = [get_shift(direction, shift) for direction, shift in zip([red_direction, green_direction, blue_direction], [red_shift, green_shift, blue_shift])]
        channels = [torch.roll(x[:, i, :, :], shifts=shifts[i], dims=(1, 2)) for i in range(3)]

        output = torch.stack(channels, dim=1)
        output = output.permute(0, 2, 3, 1)

        return (output,)

#https://github.com/hahnec/color-matcher/
from color_matcher import ColorMatcher

class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                
            },
        }
    
    CATEGORY = "postprocessing/Color Adjustments"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    
    def colormatch(self, image_ref, image_target, method):
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            out.append(torch.from_numpy(image_result))
        return (torch.stack(out, dim=0).to(torch.float32), )

class ColorCorrect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0,
                    "min": -1,
                    "max": 1,
                    "step": 0.01
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 360,
                    "step": 1
                }),
                "brightness": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 0.01
                }),
                "contrast": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 0.01
                }),
                "saturation": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 0.01
                }),
                "gamma": ("FLOAT", {
                    "default": 1,
                    "min": 0.2,
                    "max": 2.2,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"

    CATEGORY = "postprocessing/Color Adjustments"

    def color_correct(self, image: torch.Tensor, temperature: float, hue: float, brightness: float, contrast: float, saturation: float, gamma: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 2] *= 1 - temperature
            modified_image = np.clip(modified_image, 0, 255)/255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation*hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result, )

class ColorTint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "mode": (["custom","sepia", "red", "green", "blue", "cyan", "magenta", "yellow", "purple", "orange", "warm", "cool",  "lime", "navy", "vintage", "rose", "teal", "maroon", "peach", "lavender", "olive"],),
                "tint_color_hex": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_tint"

    CATEGORY = "postprocessing/Color Adjustments"

    # def hex_to_rgb(hex_color):
    #     hex_color = hex_color.lstrip('#')  # Remove the '#' character, if present
    #     r = int(hex_color[0:2], 16)
    #     g = int(hex_color[2:4], 16)
    #     b = int(hex_color[4:6], 16)
    #     return (r, g, b)

    def color_tint(self, image: torch.Tensor, strength: float, mode: str = "sepia" ,  tint_color_hex='#000000'):
        if strength == 0:
            return (image,)

      
        tint_color_hex = tint_color_hex.lstrip('#')
        

        color_rgb = [int(tint_color_hex[0:2], 16)/255,int(tint_color_hex[2:4], 16)/255,int(tint_color_hex[4:6], 16)/255]
        
      

        sepia_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 1, 3).to(image.device)

        mode_filters = {
            "custom": torch.tensor([color_rgb[0], color_rgb[1], color_rgb[2]]),
            "sepia": torch.tensor([1.0, 0.8, 0.6]),
            "red": torch.tensor([1.0, 0.6, 0.6]),
            "green": torch.tensor([0.6, 1.0, 0.6]),
            "blue": torch.tensor([0.6, 0.8, 1.0]),
            "cyan": torch.tensor([0.6, 1.0, 1.0]),
            "magenta": torch.tensor([1.0, 0.6, 1.0]),
            "yellow": torch.tensor([1.0, 1.0, 0.6]),
            "purple": torch.tensor([0.8, 0.6, 1.0]),
            "orange": torch.tensor([1.0, 0.7, 0.3]),
            "warm": torch.tensor([1.0, 0.9, 0.7]),
            "cool": torch.tensor([0.7, 0.9, 1.0]),
            "lime": torch.tensor([0.7, 1.0, 0.3]),
            "navy": torch.tensor([0.3, 0.4, 0.7]),
            "vintage": torch.tensor([0.9, 0.85, 0.7]),
            "rose": torch.tensor([1.0, 0.8, 0.9]),
            "teal": torch.tensor([0.3, 0.8, 0.8]),
            "maroon": torch.tensor([0.7, 0.3, 0.5]),
            "peach": torch.tensor([1.0, 0.8, 0.6]),
            "lavender": torch.tensor([0.8, 0.6, 1.0]),
            "olive": torch.tensor([0.6, 0.7, 0.4]),
        }

        scale_filter = mode_filters[mode].view(1, 1, 1, 3).to(image.device)

        grayscale = torch.sum(image * sepia_weights, dim=-1, keepdim=True)
        tinted = grayscale * scale_filter

        result = tinted * strength + image * (1 - strength)
        return (result,)

class FilmGrain:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "scale": ("FLOAT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "vignette": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 1.0
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "film_grain"

    CATEGORY = "postprocessing/Effects"

    def film_grain(self, image: torch.Tensor, intensity: float, scale: float, temperature: float, vignette: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Generate Perlin noise with shape (height, width) and scale
            noise = self.generate_perlin_noise((height, width), scale)
            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

            # Apply grain intensity
            noise = (noise * 2 - 1) * intensity

            # Blend the noise with the image
            grain_image = np.clip(tensor_image + noise[:, :, np.newaxis], 0, 1)

            # Apply temperature
            grain_image = self.apply_temperature(grain_image, temperature)

            # Apply vignette
            grain_image = self.apply_vignette(grain_image, vignette)

            tensor = torch.from_numpy(grain_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def generate_perlin_noise(self, shape, scale, octaves=4, persistence=0.5, lacunarity=2):
        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        def lerp(t, a, b):
            return a + t * (b - a)

        def gradient(h, x, y):
            vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
            g = vectors[h % 4]
            return g[:, :, 0] * x + g[:, :, 1] * y

        height, width = shape
        noise = np.zeros(shape)

        for octave in range(octaves):
            octave_scale = scale * lacunarity ** octave
            x = np.linspace(0, 1, width, endpoint=False)
            y = np.linspace(0, 1, height, endpoint=False)
            X, Y = np.meshgrid(x, y)
            X, Y = X * octave_scale, Y * octave_scale

            xi = X.astype(int)
            yi = Y.astype(int)

            xf = X - xi
            yf = Y - yi

            u = smoothstep(xf)
            v = smoothstep(yf)

            n00 = gradient(np.random.randint(0, 4, (height, width)), xf, yf)
            n01 = gradient(np.random.randint(0, 4, (height, width)), xf, yf - 1)
            n10 = gradient(np.random.randint(0, 4, (height, width)), xf - 1, yf)
            n11 = gradient(np.random.randint(0, 4, (height, width)), xf - 1, yf - 1)

            x1 = lerp(u, n00, n10)
            x2 = lerp(u, n01, n11)
            y1 = lerp(v, x1, x2)

            noise += y1 * persistence ** octave

        return noise / (1 - persistence ** octaves)

    def apply_temperature(self, image, temperature):
        if temperature == 0:
            return image

        temperature /= 100

        new_image = image.copy()

        if temperature > 0:
            new_image[:, :, 0] *= 1 + temperature
            new_image[:, :, 1] *= 1 + temperature * 0.4
        else:
            new_image[:, :, 2] *= 1 - temperature

        return np.clip(new_image, 0, 1)

    def apply_vignette(self, image, vignette_strength):
        if vignette_strength == 0:
            return image

        height, width, _ = image.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X ** 2 + Y ** 2)

        # Map vignette strength from 0-10 to 1.800-0.800
        mapped_vignette_strength = 1.8 - (vignette_strength - 1) * 0.1
        vignette = 1 - np.clip(radius / mapped_vignette_strength, 0, 1)

        return np.clip(image * vignette[..., np.newaxis], 0, 1)

class Glow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "blur_radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glow"

    CATEGORY = "postprocessing/Effects"

    def apply_glow(self, image: torch.Tensor, intensity: float, blur_radius: int):
        blurred_image = self.gaussian_blur(image, 2 * blur_radius + 1)
        glowing_image = self.add_glow(image, blurred_image, intensity)
        glowing_image = torch.clamp(glowing_image, 0, 1)
        return (glowing_image,)

    def gaussian_blur(self, image: torch.Tensor, kernel_size: int):
        batch_size, height, width, channels = image.shape

        sigma = (kernel_size - 1) / 6
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred.permute(0, 2, 3, 1)

        return blurred

    def add_glow(self, img, blurred_img, intensity):
        return img + blurred_img * intensity

class HSVThresholdMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
                "hsv_channel": (["hue", "saturation", "value"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hsv_threshold"

    CATEGORY = "postprocessing/Masks"

    def hsv_threshold(self, image: torch.Tensor, low_threshold: float, high_threshold: float, hsv_channel: str):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)

        if hsv_channel == "hue":
            channel = 0
            low_threshold, high_threshold = int(low_threshold * 180), int(high_threshold * 180)
        elif hsv_channel == "saturation":
            channel = 1
            low_threshold, high_threshold = int(low_threshold * 255), int(high_threshold * 255)
        elif hsv_channel == "value":
            channel = 2
            low_threshold, high_threshold = int(low_threshold * 255), int(high_threshold * 255)

        for b in range(batch_size):
            tensor_image = (image[b].numpy().copy() * 255).astype(np.uint8)
            hsv_image = cv2.cvtColor(tensor_image, cv2.COLOR_RGB2HSV)

            mask = cv2.inRange(hsv_image[:, :, channel], low_threshold, high_threshold)
            tensor = torch.from_numpy(mask).float() / 255.
            result[b] = tensor

        return (result,)

class KuwaharaBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 31,
                    "step": 1
                }),
                "method": (["mean", "gaussian"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_kuwahara_filter"

    CATEGORY = "postprocessing/Filters"

    def apply_kuwahara_filter(self, image: np.ndarray, blur_radius: int, method: str):
        if blur_radius == 0:
            return (image,)

        out = torch.zeros_like(image)
        batch_size, height, width, channels = image.shape

        for b in range(batch_size):
            image = image[b].cpu().numpy() * 255.0
            image = image.astype(np.uint8)

            out[b] = torch.from_numpy(kuwahara(image, method=method, radius=blur_radius)) / 255.0

        return (out,)

def kuwahara(orig_img, method="mean", radius=3, sigma=None):
    if method == "gaussian" and sigma is None:
        sigma = -1

    image = orig_img.astype(np.float32, copy=False)
    avgs = np.empty((4, *image.shape), dtype=image.dtype)
    stddevs = np.empty((4, *image.shape[:2]), dtype=image.dtype)
    image_2d = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY).astype(image.dtype, copy=False)
    avgs_2d = np.empty((4, *image.shape[:2]), dtype=image.dtype)

    squared_img = image_2d ** 2

    if method == "mean":
        kxy = np.ones(radius + 1, dtype=image.dtype) / (radius + 1)
    elif method == "gaussian":
        kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_32F)
        kxy /= kxy[radius:].sum()
        klr = np.array([kxy[:radius+1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]

    shift = [(0, 0), (0, radius), (radius, 0), (radius, radius)]

    for k in range(4):
        if method == "mean":
            kx, ky = kxy, kxy
        else:
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], shift[k])
        cv2.sepFilter2D(image_2d, -1, kx, ky, avgs_2d[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, stddevs[k], shift[k])
        stddevs[k] = stddevs[k] - avgs_2d[k] ** 2

    indices = np.argmin(stddevs, axis=0)
    filtered = np.take_along_axis(avgs, indices[None,...,None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)




    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "direction": (["horizontal", "vertical"],),
                "span_limit": ("INT", {
                    "default": None,
                    "min": 0,
                    "max": 100,
                    "step": 5
                }),
                "sort_by": (["hue", "saturation", "value"],),
                "order": (["forward", "backward"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sort_pixels"

    CATEGORY = "postprocessing/Effects"

    def sort_pixels(self, image: torch.Tensor, mask: torch.Tensor, direction: str, span_limit: int, sort_by: str, order: str):
        horizontal_sort = direction == "horizontal"
        reverse_sorting = order == "backward"
        sort_by = sort_by[0].upper()
        span_limit = span_limit if span_limit > 0 else None

        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_img = image[b].numpy()
            tensor_mask = mask[b].numpy()
            sorted_image = pixel_sort(tensor_img, tensor_mask, horizontal_sort, span_limit, sort_by, reverse_sorting)
            result[b] = torch.from_numpy(sorted_image)

        return (result,)

class Pixelize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pixelize"

    CATEGORY = "postprocessing/Effects"

    def apply_pixelize(self, image: torch.Tensor, pixel_size: int):
        pixelized_image = self.pixelize_image(image, pixel_size)
        pixelized_image = torch.clamp(pixelized_image, 0, 1)
        return (pixelized_image,)

    def pixelize_image(self, image: torch.Tensor, pixel_size: int):
        batch_size, height, width, channels = image.shape
        new_height = height // pixel_size
        new_width = width // pixel_size

        image = image.permute(0, 3, 1, 2)
        image = F.avg_pool2d(image, kernel_size=pixel_size, stride=pixel_size)
        image = F.interpolate(image, size=(height, width), mode='nearest')
        image = image.permute(0, 2, 3, 1)

        return image

class Quantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "dither": (["none", "floyd-steinberg"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"

    CATEGORY = "postprocessing/Color Adjustments"

    def quantize(self, image: torch.Tensor, colors: int = 256, dither: str = "FLOYDSTEINBERG"):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        dither_option = Image.Dither.FLOYDSTEINBERG if dither == "floyd-steinberg" else Image.Dither.NONE

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            palette = pil_image.quantize(colors=colors) # Required as described in https://github.com/python-pillow/Pillow/issues/5836
            quantized_image = pil_image.quantize(colors=colors, palette=palette, dither=dither_option)

            quantized_array = torch.tensor(np.array(quantized_image.convert("RGB"))).float() / 255
            result[b] = quantized_array

        return (result,)


class SineWave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amplitude": ("FLOAT", {
                    "default": 20,
                    "min": 0,
                    "max": 150,
                    "step": 5
                }),
                "frequency": ("FLOAT", {
                    "default": 10,
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
                "direction": (["horizontal", "vertical"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sine_wave"

    CATEGORY = "postprocessing/Effects"

    def apply_sine_wave(self, image: torch.Tensor, amplitude: float, frequency: float, direction: str):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b]
            result[b] = self.sine_wave_effect(tensor_image, amplitude, frequency, direction)

        return (result,)

    def sine_wave_effect(self, image: torch.Tensor, amplitude: float, frequency: float, direction: str):
        height, width, _ = image.shape
        shifted_image = torch.zeros_like(image)

        for channel in range(3):
            if direction == "horizontal":
                for i in range(height):
                    offset = int(amplitude * np.sin(2 * torch.pi * i * frequency / height))
                    shifted_image[i, :, channel] = torch.roll(image[i, :, channel], offset)
            elif direction == "vertical":
                for j in range(width):
                    offset = int(amplitude * np.sin(2 * torch.pi * j * frequency / width))
                    shifted_image[:, j, channel] = torch.roll(image[:, j, channel], offset)

        return shifted_image

class Solarize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "solarize_image"

    CATEGORY = "postprocessing/Color Adjustments"

    def solarize_image(self, image: torch.Tensor, threshold: float):
        solarized_image = torch.where(image > threshold, 1 - image, image)
        solarized_image = torch.clamp(solarized_image, 0, 1)
        return (solarized_image,)

class LensBokeh:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blades_shape": ("INT", {
                    "default": 5,
                    "min": 3,
                }),
                "blades_radius": ("INT", {
                    "default": 10,
                    "min": 1,
                }),
                "blades_rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                }),
                "blur_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "step": 2
                }),
                "blur_type": (["bilateral", "stack", "none"],),
                "method": (["dilate", "filter"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "postprocessing/lens"

    # noinspection PyUnresolvedReferences
    def lens_blur(self, image, blades_shape, blades_radius, blades_rotation, method):
        angles = np.linspace(0, 2 * np.pi, blades_shape + 1)[:-1] + blades_rotation * np.pi / 180
        x = blades_radius * np.cos(angles) + blades_radius
        y = blades_radius * np.sin(angles) + blades_radius
        pts = np.stack([x, y], axis=1).astype(np.int32)

        mask = np.zeros((blades_radius * 2 + 1, blades_radius * 2 + 1), np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        gaussian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        if method == "dilate":
            kernel = cv2.filter2D(mask, -1, gaussian_kernel)
            result = cv2.dilate(image, kernel)
        elif method == "filter":
            height, width = image.shape[:2]
            dilate_size = min(height, width) // 512

            if dilate_size > 0:
                image = cv2.dilate(image, np.ones((dilate_size, dilate_size), np.uint8))

            kernel = mask.astype(np.float32) / np.sum(mask)
            kernel = cv2.filter2D(kernel, -1, gaussian_kernel)
            result = cv2.filter2D(image, -1, kernel)
        else:
            raise ValueError("Unsupported method.")

        return result

    def node(self, images, blades_shape, blades_radius, blades_rotation, blur_size, blur_type, method):
        tensor = images.clone().detach()
        blur_size -= 1

        if blur_type == "bilateral":
            tensor = cv2_layer(tensor, lambda x: cv2.bilateralFilter(x, blur_size, -100, 100))
        elif blur_type == "stack":
            tensor = cv2_layer(tensor, lambda x: cv2.stackBlur(x, (blur_size, blur_size)))
        elif blur_type == "none":
            pass
        else:
            raise ValueError("Unsupported blur type.")

        return (cv2_layer(tensor, lambda x: self.lens_blur(
            x, blades_shape, blades_radius, blades_rotation, method
        )),)

class LensOpticAxis:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lens_shape": (["circle", "square", "rectangle", "corners"],),
                "lens_edge": (["around", "symmetric"],),
                "lens_curvy": ("FLOAT", {
                    "default": 4.0,
                    "max": 15.0,
                    "step": 0.1,
                }),
                "lens_zoom": ("FLOAT", {
                    "default": 2.0,
                    "step": 0.1,
                }),
                "lens_aperture": ("FLOAT", {
                    "default": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "blur_intensity": ("INT", {
                    "default": 30,
                    "min": 2,
                    "step": 2
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "node"
    CATEGORY = "postprocessing/lens"

    def node(self, images, lens_shape, lens_edge, lens_curvy, lens_zoom, lens_aperture, blur_intensity):
        blur_intensity -= 1
        lens_zoom += 1

        height, width = images[0, :, :, 0].shape

        if lens_edge == "around":
            mask = radialspace_2D((height, width), lens_curvy, lens_zoom, lens_shape, 0.0, 1.0 + lens_curvy).unsqueeze(0).unsqueeze(3)
        elif lens_edge == "symmetric":
            if height != width:
                new_height = new_width = max(height, width)
                crop_top_bottom = (new_height - height) // 2
                crop_left_right = (new_width - width) // 2

                mask = radialspace_2D((new_height, new_width), lens_curvy, lens_zoom, lens_shape, 0.0, 1.0 + lens_curvy)[
                   crop_top_bottom:-crop_top_bottom or None,
                   crop_left_right:-crop_left_right or None
                ].unsqueeze(0).unsqueeze(3)
            else:
                mask = radialspace_2D((height, width), lens_curvy, lens_zoom, lens_shape, 0.0, 1.0 + lens_curvy).unsqueeze(0).unsqueeze(3)
        else:
            raise ValueError("Not existing lens_edge parameter.")

        center_x = width // 2
        center_y = height // 2

        y_v, x_v = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        dx = x_v - center_x
        dy = y_v - center_y

        distance = torch.sqrt(dx ** 2 + dy ** 2)

        map_x = x_v + mask[0, :, :, 0] * dx / distance * (-lens_aperture * 100)
        map_y = y_v + mask[0, :, :, 0] * dy / distance * (-lens_aperture * 100)

        map_x = map_x.to(torch.float32).numpy()
        map_y = map_y.to(torch.float32).numpy()

        shifted_images = cv2_layer(images, lambda x: cv2.remap(x, map_x, map_y, cv2.INTER_LINEAR))
        shifted_mask = cv2_layer(mask, lambda x: cv2.remap(x, map_x, map_y, cv2.INTER_LINEAR))
        edited_images = cv2_layer(shifted_images, lambda x: cv2.stackBlur(x, (blur_intensity, blur_intensity)))

        mask = torch.clamp(mask, 0.0, 1.0)
        shifted_mask = torch.clamp(shifted_mask, 0.0, 1.0)

        result = shifted_images * (1 - mask) + edited_images * mask

        return result, shifted_mask

class LensZoomBurst:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "step": 0.01
                }),
                "samples": ("INT", {
                    "default": 100,
                    "min": 1,
                }),
                "position_x": ("FLOAT", {
                    "default": 0.5,
                    "max": 1.0,
                    "step": 0.01
                }),
                "position_y": ("FLOAT", {
                    "default": 0.5,
                    "max": 1.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                }),
                "method": (["circle", "point"],),
                "stabilization": (["true", "false"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "postprocessing/lens"

    # noinspection PyUnresolvedReferences
    def zoom_burst(
            self,
            image,
            scale,
            samples,
            position,
            rotation,
            method,
            stabilization,
    ):
        if scale < 1.0:
            raise ValueError("Parameter scale can't be smaller then initial image size.")

        h, w = image.shape[:2]

        x = np.arange(w)
        y = np.arange(h)

        xx, yy = np.meshgrid(x, y)

        cx = int(w * position[0])
        cy = int(h * position[1])

        if method == "circle":
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            max_d = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

            map_x_up = (xx - d * (scale - 1.0) / max_d * (xx - cx) / samples).astype(np.float32)
            map_y_up = (yy - d * (scale - 1.0) / max_d * (yy - cy) / samples).astype(np.float32)

            map_x_down = (xx + d * (scale - 1.0) / max_d * (xx - cx) / samples).astype(np.float32)
            map_y_down = (yy + d * (scale - 1.0) / max_d * (yy - cy) / samples).astype(np.float32)
        elif method == "point":
            map_x_up = (xx - (xx - cx) * (scale - 1.0) / samples).astype(np.float32)
            map_y_up = (yy - (yy - cy) * (scale - 1.0) / samples).astype(np.float32)

            map_x_down = (xx + (xx - cx) * (scale - 1.0) / samples).astype(np.float32)
            map_y_down = (yy + (yy - cy) * (scale - 1.0) / samples).astype(np.float32)
        else:
            raise ValueError("Unsupported method.")

        if rotation > 0.0:
            angle_step = rotation / samples

            rm_up = cv2.getRotationMatrix2D((cx, cy), angle_step, 1)
            rm_down = cv2.getRotationMatrix2D((cx, cy), -angle_step, 1)
        else:
            vibration_angle = 1.0
            vibration_step = vibration_angle / samples

            rm_up_even = cv2.getRotationMatrix2D((cx, cy), vibration_step, 1)
            rm_down_even = cv2.getRotationMatrix2D((cx, cy), -vibration_step, 1)

            rm_up_odd = cv2.getRotationMatrix2D((cx, cy), -vibration_step, 1)
            rm_down_odd = cv2.getRotationMatrix2D((cx, cy), vibration_step, 1)

        for i in range(samples):
            if stabilization:
                tmp_up = cv2.remap(image, map_x_up, map_y_up, cv2.INTER_LINEAR)
                tmp_down = cv2.remap(image, map_x_down, map_y_down, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                if rotation > 0.0:
                    tmp_up = cv2.warpAffine(tmp_up, rm_up, (w, h), borderMode=cv2.BORDER_REFLECT)
                    tmp_down = cv2.warpAffine(tmp_down, rm_down, (w, h), borderMode=cv2.BORDER_REFLECT)
                else:
                    if i % 2 == 0:
                        tmp_up = cv2.warpAffine(tmp_up, rm_up_even, (w, h), borderMode=cv2.BORDER_REFLECT)
                        tmp_down = cv2.warpAffine(tmp_down, rm_down_even, (w, h), borderMode=cv2.BORDER_REFLECT)
                    else:
                        tmp_up = cv2.warpAffine(tmp_up, rm_up_odd, (w, h), borderMode=cv2.BORDER_REFLECT)
                        tmp_down = cv2.warpAffine(tmp_down, rm_down_odd, (w, h), borderMode=cv2.BORDER_REFLECT)

                image = cv2.addWeighted(tmp_up, 0.5, tmp_down, 0.5, 0)
            else:
                tmp = cv2.remap(image, map_x_up, map_y_up, cv2.INTER_LINEAR)

                if rotation > 0.0:
                    rm = cv2.getRotationMatrix2D((cx, cy), angle_step, 1)
                    tmp = cv2.warpAffine(tmp, rm, (w, h), borderMode=cv2.BORDER_REFLECT)
                else:
                    if i % 2 == 0:
                        tmp = cv2.warpAffine(tmp, rm_up_even, (w, h), borderMode=cv2.BORDER_REFLECT)
                    else:
                        tmp = cv2.warpAffine(tmp, rm_up_odd, (w, h), borderMode=cv2.BORDER_REFLECT)

                image = cv2.addWeighted(tmp, 0.5, image, 0.5, 0)

        return image

    def node(self, images, scale, samples, position_x, position_y, rotation, method, stabilization):
        tensor = images.clone().detach()

        return (cv2_layer(tensor, lambda x: self.zoom_burst(
            x, scale, samples, (position_x, position_y), rotation, method, True if stabilization == "true" else False
        )),)

class LUT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "lut_image": ("IMAGE",),
                             "gamma_correction": (['True','False'],)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'apply_lut'
    CATEGORY = 'Processing/color adjust'
    OUTPUT_NODE = True
    
    

    @apply_to_batch
    def apply_lut(self, image, lut_image, gamma_correction):
        img = tensor2pil(image)
        lut_image = tensor2pil(lut_image)
        
        if gamma_correction == 'True':
            corrected_img = gamma_correction_pil(img, 1.0/2.2)
        else:
            corrected_img = img
        filtered_image = apply_hald_clut(lut_image, corrected_img).convert("RGB")
        #return (pil2tensor(filtered_image), )
        return pil2tensor(filtered_image)

    @classmethod
    def IS_CHANGED(self, hald_clut):
        return (np.nan,)

from typing import Literal, Any
class Vignette:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lens_shape": (["circle", "rectangle"],),
                "lens_edge": (["around", "symmetric"],),
                "lens_curvy": ("FLOAT", {
                    "default": 3.0,
                    "max": 15.0,
                    "step": 0.1,
                }),
                "lens_zoom": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.1,
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "saturation": ("FLOAT", {
                    "default": 0.5,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "node"
    CATEGORY = "postprocessing/Effects"

    def node(self, images, lens_shape, lens_edge, lens_curvy, lens_zoom, brightness, saturation):
        tensor = images.clone().detach()

        lens_zoom += 1

        height, width = tensor[0, :, :, 0].shape

        if lens_edge == "around":
            mask = radialspace_2D((height, width), lens_curvy, lens_zoom, lens_shape).unsqueeze(0).unsqueeze(3)
        elif lens_edge == "symmetric":
            if height != width:
                new_height = new_width = max(height, width)
                crop_top_bottom = (new_height - height) // 2
                crop_left_right = (new_width - width) // 2

                mask = radialspace_2D((new_height, new_width), lens_curvy, lens_zoom, lens_shape)[
                       crop_top_bottom:-crop_top_bottom or None,
                       crop_left_right:-crop_left_right or None
                ].unsqueeze(0).unsqueeze(3)
            else:
                mask = radialspace_2D((height, width), lens_curvy, lens_zoom, lens_shape).unsqueeze(0).unsqueeze(3)
        else:
            raise ValueError("Not existing lens_edge parameter.")

        tensor = tensor.permute(0, 3, 1, 2)
        tensor[:, 0:3, :, :] = D.adjust_brightness(tensor[:, 0:3, :, :], brightness)
        tensor[:, 0:3, :, :] = D.adjust_saturation(tensor[:, 0:3, :, :], saturation)
        tensor = tensor.permute(0, 2, 3, 1)

        result = images * (1 - mask) + tensor * mask

        mask = mask.squeeze()

        return result, mask

def gaussian_kernel(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()

def sort_span(span, sort_by, reverse_sorting):
    if sort_by == 'H':
        key = lambda x: x[1][0]
    elif sort_by == 'S':
        key = lambda x: x[1][1]
    else:
        key = lambda x: x[1][2]

    span = sorted(span, key=key, reverse=reverse_sorting)
    return [x[0] for x in span]


def find_spans(mask, span_limit=None):
    spans = []
    start = None
    for i, value in enumerate(mask):
        if value == 0 and start is None:
            start = i
        if value == 1 and start is not None:
            span_length = i - start
            if span_limit is None or span_length <= span_limit:
                spans.append((start, i))
            start = None
    if start is not None:
        span_length = len(mask) - start
        if span_limit is None or span_length <= span_limit:
            spans.append((start, len(mask)))

    return spans


def pixel_sort(img, mask, horizontal_sort=False, span_limit=None, sort_by='H', reverse_sorting=False):
    height, width, _ = img.shape
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[..., 0] /= 2.0  # Scale H channel to [0, 1] range

    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # loop over the rows and replace contiguous bands of 1s
    for i in range(height if horizontal_sort else width):
        in_band = False
        start = None
        end = None
        for j in range(width if horizontal_sort else height):
            if (mask[i, j] if horizontal_sort else mask[j, i]) == 1:
                if not in_band:
                    in_band = True
                    start = j
                end = j
            else:
                if in_band:
                    for k in range(start+1, end):
                        if horizontal_sort:
                            mask[i, k] = 0
                        else:
                            mask[k, i] = 0
                    in_band = False

        if in_band:
            for k in range(start+1, end):
                if horizontal_sort:
                    mask[i, k] = 0
                else:
                    mask[k, i] = 0

    sorted_image = np.zeros_like(img)
    if horizontal_sort:
        for y in range(height):
            row_mask = mask[y]
            spans = find_spans(row_mask, span_limit)
            sorted_row = np.copy(img[y])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for x in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_row[start + i] = pixel
            sorted_image[y] = sorted_row
    else:
        for x in range(width):
            column_mask = mask[:, x]
            spans = find_spans(column_mask, span_limit)
            sorted_column = np.copy(img[:, x])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for y in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_column[start + i] = pixel
            sorted_image[:, x] = sorted_column

    return sorted_image

NODE_CLASS_MAPPINGS = {
    "MX_Blend": Blend,
    "MX_AlphaBlend": AlphaBlend,
    "MX_Blur": Blur,
    "MX_Canny": CannyEdgeMask,
    "MX_ColorMatch": ColorMatch,
    "MX_ChromaticAberration": ChromaticAberration,
    "MX_ColorCorrect": ColorCorrect,
    "MX_ColorTint": ColorTint,
    "MX_Noise": FilmGrain,
    "MX_Glow": Glow,
    "MX_HSVThresholdMask": HSVThresholdMask,
    "MX_KuwaharaBlur(Cartoon)": KuwaharaBlur,
    "MX_Mosaic": Pixelize,
    "MX_Posterize": Quantize,
    "MX_SineWave": SineWave,
    "MX_Solarize": Solarize,
    "MX_LensBokeh": LensBokeh,
    "MX_LensOpticAxis": LensOpticAxis,
    "MX_LensZoomBurst": LensZoomBurst,
    "MX_LUT":LUT,
    "MX_Vignette": Vignette,
}
