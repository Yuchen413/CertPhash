import torch
import torchvision
import torchvision.transforms.functional as TF
from imgaug.augmenters.arithmetic import JpegCompression as ImgAugJpegCompression
import torchvision.transforms as transforms

class PepperSaltNoise:
    """Applies salt-and-pepper noise with specified parameters."""

    def __init__(self, level=0.01):
        """
        Args:
            level (float): Level of noise (probability of a pixel being altered). Value should be between 0 and 1.
        """
        self.level = level

    def __call__(self, img):
        """
        Apply salt-and-pepper noise to the input image.

        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            Tensor: Image with salt-and-pepper noise.
        """
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)

        # Generate random noise mask for salt and pepper
        noise = torch.rand(img.shape)

        # Apply salt noise (white pixels)
        img[noise < (self.level / 2)] = 1.0

        # Apply pepper noise (black pixels)
        img[noise > 1 - (self.level / 2)] = 0.0

        return img

class RandomErasing:
    """Applies random erasing with specified parameters."""

    def __init__(self, scale = 0.3):
        """
        Args:
            p (float): Probability of applying the transformation.
            scale (tuple): Range of roportion of erased area against the input image area.
            value (int, float, or tuple): Erasing value. 0 for black, or a random value if set to 'random'.
        """
        self.transform = transforms.RandomErasing(p=1, scale=scale, value=0)

    def __call__(self, x):
        return self.transform(x)

class ResizeTransform:
    """Resizes the image to a specified size."""

    def __init__(self, size):
        """
        Args:
            size (int, tuple): The target size. If int, the smaller edge of the image is matched to this size while preserving aspect ratio.
                               If tuple (height, width), the image is resized to the exact dimensions.
        """
        self.transform = transforms.Resize(size)

    def __call__(self, x):
        return self.transform(x)


class Grayscale:
    """Converts an image to grayscale with specified probability."""

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of applying the transformation.
        """
        self.transform = transforms.RandomGrayscale(p=p)

    def __call__(self, x):
        return self.transform(x)


class GaussianBlur:
    """Applies Gaussian blur with specified parameters."""

    def __init__(self, sigma=0.5):
        """
        Args:
            kernel_size (int or tuple): Size of the Gaussian kernel.
            sigma (tuple): Range of standard deviation for the Gaussian kernel.
        """
        self.transform = transforms.GaussianBlur(kernel_size=5, sigma=sigma)

    def __call__(self, x):
        return self.transform(x)


class RandomPerspectiveTransform:
    """Applies random perspective transformation with specified parameters."""

    def __init__(self, distortion_scale=0.5, p=1):
        """
        Args:
            distortion_scale (float): Controls the degree of distortion. Higher values result in stronger perspective changes.
            p (float): Probability of applying the transformation.
        """
        self.transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def __call__(self, x):
        return self.transform(x)


class Rotate:
    """Rotates by the given angle"""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        if self.angle == 0:
            return x
        return TF.rotate(x, self.angle)


class Translate:
    """Translates by the given number of pixels in vertical and horizontal direction."""

    def __init__(self, translation: (float, float)):
        self.horizontal = translation[0]
        self.vertical = translation[1]

    def __call__(self, x):
        return TF.affine(
            x,
            angle=0,
            translate=[self.horizontal, self.vertical],
            scale=1.0,
            shear=[0, 0]
        )


class ChangeHue:
    """Changes the hue of an image"""

    def __init__(self, hue_angle: int):
        """
        :param hue_value: The angle by which the hue of the image is shifted. Must be in [-180, 180]. -180 and 180 will
        yield the same results which is an image with complementary colors.
        """
        self.hue_angle = hue_angle

    def __call__(self, x):
        if self.hue_angle == 0:
            return x
        return TF.adjust_hue(x, self.hue_angle / 360)


class ChangeBrightness:
    """Changes the brightness of an image"""

    def __init__(self, brightness_factor: float):
        """
        :param brightness_factor: Factor by which the brightness is changed. 0 yields a black image while 2 doubles
        the brightness of the image. (1 yields original image).
        """
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        if self.brightness_factor == 1:
            return x
        return TF.adjust_brightness(x, self.brightness_factor)


class ChangeContrast:
    """Changes the contrast of an image"""

    def __init__(self, contrast_factor: float):
        """
        :param contrast_factor: Factor by which the contrast is adjusted. 0 yields gray image while 1 gives the original
        image and 2 doubles the contrast.
        """
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        if self.contrast_factor == 1:
            return x
        return TF.adjust_contrast(x, self.contrast_factor)


class ChangeSaturation:
    """Changes the saturation of an image"""

    def __init__(self, saturation_factor: float):
        """
        :param saturation_factor: Factor by which the saturation is adjusted. 0 yields black and white image while 1
        gives the original image and 2 doubles the saturation.
        """
        self.saturation_factor = saturation_factor

    def __call__(self, x):
        if self.saturation_factor == 1:
            return x
        return TF.adjust_saturation(x, self.saturation_factor)


class JpegCompression:
    """Compresses a given image using the JPEG algorithm"""

    def __init__(self, compression_value: int):
        """
        :param compression_value: Degree of compression. Has to be in range [0, 100] where 0 is no compression and 100
        is maximum compression.
        """
        self.compression = compression_value
        self.compression_aug = ImgAugJpegCompression(compression=self.compression, seed=42)
        self.target = ''

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.denormalize = torchvision.transforms.Normalize((-1 * mean / std), (1.0 / std))

    def __call__(self, x):
        if 'nn' in self.target:
            x = self.denormalize(x)
        input_img = (x.permute(1, 2, 0).numpy() * 255).astype('uint8')
        compressed_img = self.compression_aug(image=input_img)

        if self.compression == 0:
            return x
        return torch.tensor(compressed_img / 255, dtype=torch.float).permute(2, 0, 1)


class HorizontalFlipping:
    """Flips a given image horizontally"""

    def __init__(self):
        pass

    def __call__(self, x):
        return TF.hflip(x)


class VerticalFlipping:
    """Flips a given image vertically"""

    def __init__(self):
        pass

    def __call__(self, x):
        return TF.vflip(x)


class BlackBorder:
    """Adds a black border to the given image while keeping the overall size the same. This means that the image is
    shrunk to create the black border. Expects the image to be the shape CxHxW"""

    def __init__(self, image_size: int):
        """
        :param image_size: The size of the border in pixels.
        """
        self.image_size = image_size

    def __call__(self, x: torch.Tensor):
        if self.image_size == 0:
            return x

        new_x = TF.resize(x, size=[self.image_size, self.image_size])
            
        # create an empty tensor with added borders
        new_tensor = torch.zeros_like(x)

        # assign the image
        img_yindex_start = int((x.shape[1] - self.image_size) / 2)
        img_yindex_end = -img_yindex_start
        img_xindex_start = int((x.shape[2] - self.image_size) / 2)
        img_xindex_end = -img_xindex_start
        if img_yindex_end == 0:
            img_yindex_end = None
        if img_xindex_end == 0:
            img_xindex_end = None
        new_tensor[:, img_yindex_start:img_yindex_end, img_xindex_start:img_xindex_end] = new_x

        # resize the image to the original size
        return new_tensor


class CenterCrop:
    """Crops to the center of the image."""

    def __init__(self, crop_size: int):
        """
        :param crop_size: The size of the crop region
        """
        self.crop_size = crop_size

    def __call__(self, x):
        if self.crop_size == 64:
            return x
        cropped_region = TF.center_crop(x, output_size=self.crop_size)
        return TF.resize(cropped_region, size=[x.shape[1], x.shape[2]])
