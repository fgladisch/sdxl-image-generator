import os
import platform
import subprocess
from datetime import datetime

import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

load_dotenv()


OUTPUT_FOLDER = "output"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_MODEL = "ByteDance/SDXL-Lightning"
INFERENCE_STEPS = 8
MODEL_CHECKPOINTS = f"sdxl_lightning_{INFERENCE_STEPS}step_unet.safetensors"

prompt_text = os.getenv("PROMPT", "")


def get_device():
    # check for nvidia cuda support
    if torch.cuda.is_available():
        return torch.device("cuda")

    # check for mac support
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


torch_device = get_device()
torch_dtype = torch.float16 if torch_device != torch.device("cpu") else torch.float32

# Load model.
unet_config = UNet2DConditionModel.load_config(BASE_MODEL, subfolder="unet")
unet = UNet2DConditionModel.from_config(unet_config).to(torch_device, torch_dtype)
unet.load_state_dict(load_file(hf_hub_download(UNET_MODEL, MODEL_CHECKPOINTS)))

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, unet=unet, torch_dtype=torch_dtype, variant="fp16"
).to(torch_device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# create output folder if not exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# use datetime to create a unique filename
filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = os.path.join(OUTPUT_FOLDER, f"{filename}.png")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe(prompt_text, num_inference_steps=INFERENCE_STEPS, guidance_scale=0).images[0].save(
    filepath
)

system_name = platform.system()

print(f"Opening {filepath} on {system_name}...")

if system_name == "Darwin":  # macOS
    subprocess.call(("open", filepath))
elif system_name == "Windows":  # Windows
    os.startfile(filepath)
else:  # linux variants
    subprocess.call(("xdg-open", filepath))
