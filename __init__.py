"""SeedVR2 Image Upscaler - ComfyUI custom node for memory-efficient image upscaling."""

import os
from PIL import Image
from .upscaler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Large stitched outputs can exceed Pillow's decompression-bomb threshold
# for legitimate upscaling jobs.
#
# Default behavior here is unlimited pixels.
# To enforce a limit, explicitly set:
#   SEEDVR2_MAX_IMAGE_PIXELS=<int>   (for example: 500000000)
# To force-disable checks via env value:
#   SEEDVR2_MAX_IMAGE_PIXELS=none
_max_pixels_env = os.getenv("SEEDVR2_MAX_IMAGE_PIXELS", "").strip()
if not _max_pixels_env:
    Image.MAX_IMAGE_PIXELS = None
else:
    if _max_pixels_env.lower() in {"none", "disable", "unlimited", "0"}:
        Image.MAX_IMAGE_PIXELS = None
    else:
        try:
            parsed_limit = int(_max_pixels_env)
            Image.MAX_IMAGE_PIXELS = parsed_limit if parsed_limit > 0 else None
        except ValueError:
            # Invalid env value: do not enforce a limit.
            Image.MAX_IMAGE_PIXELS = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
