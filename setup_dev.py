import site
from pathlib import Path

root = Path.cwd()
pth = Path(site.getsitepackages()[0]) / "local_comfy_paths.pth"
pth.write_text(
    "\n".join([
        str(root / "ComfyUI"),
        str(root / "ComfyUI-SeedVR2_VideoUpscaler"),
        str(root / "ComfyUI-SeedVR2_VideoUpscaler" / "src"),  # needed for imports like from src...
    ]) + "\n",
    encoding="utf-8",
)
print("Wrote:", pth)
