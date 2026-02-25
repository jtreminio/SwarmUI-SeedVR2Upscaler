"""Main upscaler class for ComfyUI SeedVR2 tiling upscaling node.

This node wraps the SeedVR2 VideoUpscaler with intelligent tiling for
memory-efficient processing of large images.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from .progress import Progress
from .image_utils import ImageUtils
from .tiling import TileUtils
from .stitching import StitchingPipeline


def _get_offload_device_options() -> list[str]:
    """Build offload device options similar to upstream VideoUpscaler node."""
    devices = ["none", "cpu"]

    if torch.cuda.is_available():
        devices.extend(f"cuda:{idx}" for idx in range(torch.cuda.device_count()))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    return devices


def _debug_log(enabled: bool, message: str) -> None:
    """Emit extension-specific debug logs only when enabled."""
    if enabled:
        print(f"[SeedVR2 Tiling][debug] {message}", flush=True)


def _resolve_tile_upscale_resolution(tile_upscale_resolution: int, tile_size: int, upscale_factor: float) -> int:
    """Resolve tile upscale resolution, supporting 0 as auto-infer mode."""
    min_resolution = 64
    max_resolution = 8192
    step = 8

    requested = int(tile_upscale_resolution)
    if requested <= 0:
        inferred = int(round(tile_size * upscale_factor))
        requested = inferred

    requested = max(min_resolution, min(max_resolution, requested))
    requested = int(round(requested / step) * step)
    requested = max(min_resolution, min(max_resolution, requested))

    return requested


class SeedVR2ImageUpscaler:
    """Tiled upscaling node that wraps SeedVR2 for memory-efficient processing."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "dit": ("SEEDVR2_DIT", {
                    "tooltip": "DiT model configuration from 'SeedVR2 (Down)Load DiT Model' node."
                }),
                "vae": ("SEEDVR2_VAE", {
                    "tooltip": "VAE model configuration from 'SeedVR2 (Down)Load VAE Model' node."
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "Random seed for reproducible results. Same seed produces same output."
                }),
                "resolution": ("INT", {
                    "default": 1080,
                    "min": 16,
                    "max": 16384,
                    "step": 2,
                    "tooltip": "Target resolution for the smallest output side. The larger side is scaled proportionally."
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Square tile size in pixels (applied to both width and height). Smaller tiles use less VRAM but may show more seams."
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Tile edge blending. 0=multi-band frequency separation (best detail), 1-3=minimal blur, 4+=traditional blur."
                }),
                "tile_overlap": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Overlap between tiles in pixels. Higher values reduce seams but increase processing time. Recommended: 32-64."
                }),
                "tile_upscale_resolution": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Resolution for upscaling each tile. Set 0 to auto-infer from tile_size and output scale. Higher=better quality but more VRAM."
                }),
                "tiling_strategy": (["Chess", "Linear"], {
                    "tooltip": "Tile processing order. Chess=checkerboard pattern for better blending, Linear=row-by-row (faster)."
                }),
                "anti_aliasing_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Edge-aware smoothing strength. 0=disabled, 0.1-0.3=subtle smoothing. May soften details."
                }),
                "blending_method": (["auto", "multiband", "bilateral", "content_aware", "linear", "simple"], {
                    "default": "auto",
                    "tooltip": "Blending algorithm: auto (mask_blur based), multiband (Laplacian pyramid/frequency separation), bilateral (edge-preserving filter), content_aware (structure-adaptive), linear (alpha blend), simple (pixel averaging)."
                }),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {
                    "default": "lab",
                    "tooltip": "Color correction method to match upscaled output to original input colors. lab=perceptual matching (recommended), wavelet=frequency-based, wavelet_adaptive=with saturation correction, hsv=hue-conditional, adain=style transfer, none=disabled."
                }),
                "input_noise_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Input noise injection scale (default: 0.0). Can help with artifact reduction on some images."
                }),
                "offload_device": (_get_offload_device_options(), {
                    "default": "cpu",
                    "tooltip": "Device for storing intermediate tensors between phases. none=fastest/highest VRAM, cpu=lower VRAM usage, cuda:X=another GPU."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed tiling-node diagnostics (tile layout, batching, blend strategy, and timing)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        image: torch.Tensor,
        dit: dict[str, Any],
        vae: dict[str, Any],
        seed: int,
        resolution: int,
        tile_size: int,
        mask_blur: int,
        tile_overlap: int,
        tile_upscale_resolution: int,
        tiling_strategy: str,
        anti_aliasing_strength: float,
        blending_method: str = "auto",
        color_correction: str = "lab",
        input_noise_scale: float = 0.0,
        offload_device: str = "cpu",
        enable_debug: bool = False,
    ) -> tuple[torch.Tensor]:
        try:
            start_time = time.perf_counter()
            progress: Progress | None = None

            # Setup
            pil_image = ImageUtils.tensor_to_pil(image)
            upscale_factor = resolution / min(pil_image.width, pil_image.height)
            output_width = int(pil_image.width * upscale_factor)
            output_height = int(pil_image.height * upscale_factor)
            resolved_tile_upscale_resolution = _resolve_tile_upscale_resolution(
                tile_upscale_resolution, tile_size, upscale_factor
            )

            # Generate tiles and update progress tracker with correct count
            main_tiles = TileUtils.generate_tiles(pil_image, tile_size, tile_overlap, tiling_strategy)
            progress = Progress(len(main_tiles), enable_debug=enable_debug)
            progress.initialize_websocket_progress()

            _debug_log(
                enable_debug,
                (
                    f"Input={pil_image.width}x{pil_image.height}, Output={output_width}x{output_height}, "
                    f"Tiles={len(main_tiles)}, TileSize={tile_size}x{tile_size}, "
                    f"Overlap={tile_overlap}, Strategy={tiling_strategy}, Blend={blending_method}, "
                    f"TileUpscale={resolved_tile_upscale_resolution}"
                    f"{' (auto)' if int(tile_upscale_resolution) <= 0 else ''}, "
                    f"InputNoise={input_noise_scale:.4f}, Offload={offload_device}"
                ),
            )

            # Process and stitch tiles
            output_image = StitchingPipeline.process_and_stitch(
                tiles=main_tiles,
                width=output_width,
                height=output_height,
                dit_config=dit,
                vae_config=vae,
                seed=seed,
                tile_upscale_resolution=resolved_tile_upscale_resolution,
                upscale_factor=upscale_factor,
                mask_blur=mask_blur,
                progress=progress,
                original_image=pil_image,
                anti_aliasing_strength=anti_aliasing_strength,
                blending_method=blending_method,
                color_correction=color_correction,
                input_noise_scale=input_noise_scale,
                offload_device=offload_device,
                enable_debug=enable_debug,
            )

            # Finalize progress
            progress.finalize_websocket_progress()
            _debug_log(enable_debug, f"Completed tiling upscale in {time.perf_counter() - start_time:.2f}s")

            return (ImageUtils.pil_to_tensor(output_image),)

        except Exception as e:
            # Ensure progress is completed even on error
            if progress is not None:
                progress.finalize_websocket_progress()
            raise e


NODE_CLASS_MAPPINGS = {
    "SeedVR2ImageUpscaler": SeedVR2ImageUpscaler,
    "SeedVR2TilingUpscaler": SeedVR2ImageUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2ImageUpscaler": "SeedVR2 Image Upscaler",
    "SeedVR2TilingUpscaler": "SeedVR2 Image Upscaler",
}
