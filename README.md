#  ComfyUI-MX-post-processing-nodes

A collection of post processing nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), dds image post-processing adjustment capabilities to the ComfyUI.

## Example Image and Workflow

<p align="center">
  <img src="examples/MX_postprocessing_example.jpg" width="50%" />
 
</p>



---

<details>
	<summary>$\Large\color{#00A7B5}\text{Expand Node List}$</summary>

<br/>

 - MX_Blend: Blends two images using arithmetic operations like add,multiply, overlay, darken,lighten.......
 - MX_AlphaBlend: Blends two images alpha mask
 - MX_Blur: Applies a Gaussian blur to the input image, softening the details
 - MX_CannyEdgeMask: Creates a mask using canny edge detection
 - MX_Chromatic Aberration: Shifts the color channels in an image, creating a glitch aesthetic
 - MX_ColorCorrect: Adjusts the color balance, temperature, hue, brightness, contrast, saturation, and gamma of an image
 - MX_ColorTint: Applies a customizable tint to the input image, with various color modes such as sepia, RGB, CMY and several composite colors
 - MX_FilmGrain: Adds a film grain effect to the image, along with options to control the temperature, and vignetting.
 - MX_Glow: Applies a blur with a specified radius and then blends it with the original image. Creates a nice glowing effect.
 - MX_HSVThresholdMask: Creates a mask by thresholding HSV (hue, saturation, and value) channels
 - MX_KuwaharaBlur:Applies an edge preserving blur, creating a more realistic blur than Gaussian.
 - MX_PixelSort:Rearranges the pixels in the input image based on their values, and input mask. Creates a cool glitch like effect.
 - MX_Pixelize: Applies a pixelization effect, simulating the reducing of resolution
 - MX_Quantize: Set and dither the amount of colors in an image from 0-256, reducing color information
 - MX_Sharpen: Enhances the details in an image by applying a sharpening filter
 - MX_SineWave: Runs a sine wave through the image, making it appear squiggly
 - MX_Solarize: Inverts image colors based on a threshold for a striking, high-contrast effect
 - MX_Vignette: Applies a vignette effect, putting the corners of the image in shadow



</details>

---

## Install

To install these nodes 

  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - Run `git clone https://github.com/Intersection98/MX-ComfyUI-post-processing-nodes/`
  - pip install -r requirements.txt



## Reference
https://github.com/EllangoK/ComfyUI-post-processing-nodes/
https://github.com/digitaljohn/comfyui-propost


