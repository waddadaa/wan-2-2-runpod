"""
WAN 2.2 Text+Image-to-Video (TI2V) Handler for RunPod Serverless
Model: Wan2.2-TI2V-5B (5B parameters)
GPU: RTX 4090 24GB or A100
License: Apache 2.0 (Commercial use allowed)
"""

import os
import sys
import gc
import torch
import base64
import tempfile
import traceback
import logging
from typing import Dict, Any
from PIL import Image
from io import BytesIO

import runpod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix safetensors "device cuda:0 is invalid" error (Rust CUDA bug workaround)
import safetensors.torch
_original_load_file = safetensors.torch.load_file

def _patched_load_file(filename, device="cpu"):
    result = _original_load_file(filename, device="cpu")
    if device != "cpu" and torch.cuda.is_available():
        target = torch.device("cuda:0")
        result = {k: v.to(target) for k, v in result.items()}
    return result

safetensors.torch.load_file = _patched_load_file
logger.info("Patched safetensors to load via CPU")

# Add WAN 2.2 to path
sys.path.insert(0, "/app/Wan2.2")

MODEL = None

# Model configuration
MODEL_NAME = "Wan2.2-TI2V-5B"
HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_CONFIG_KEY = "ti2v-5B"


def ensure_model_downloaded(model_dir: str, model_name: str, hf_repo_id: str) -> str:
    """Check if model exists, download from HuggingFace if not."""
    ckpt_dir = os.path.join(model_dir, model_name)

    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        logger.info(f"Model found at {ckpt_dir}")
        return ckpt_dir

    logger.info("=" * 60)
    logger.info(f"Model not found at {ckpt_dir}")
    logger.info(f"Downloading {model_name} from HuggingFace...")
    logger.info(f"Repo: {hf_repo_id}")
    logger.info("This may take a while (~33GB download)")
    logger.info("=" * 60)

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"Model downloaded successfully to {ckpt_dir}")
        return ckpt_dir
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Failed to download model from {hf_repo_id}: {e}")


def load_model():
    """Load TI2V-5B model, downloading from HuggingFace if needed."""
    global MODEL
    if MODEL is not None:
        return MODEL

    # Initialize CUDA before loading model
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        _ = torch.zeros(1).cuda()
        logger.info(f"CUDA initialized: {torch.cuda.get_device_name(0)}")

    model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models")
    ckpt_dir = ensure_model_downloaded(model_dir, MODEL_NAME, HF_REPO_ID)

    logger.info("=" * 60)
    logger.info("Loading WAN 2.2 TI2V-5B model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[WAN_CONFIG_KEY]

        MODEL = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("TI2V-5B model loaded successfully!")
        logger.info("=" * 60)
        return MODEL
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def decode_base64_image(base64_string: str) -> bytes:
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    return base64.b64decode(base64_string)


def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_temp_file(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


def parse_resolution(size: str) -> tuple:
    if not size:
        return (832, 480)
    if "*" in size:
        parts = size.split("*")
    elif "x" in size.lower():
        parts = size.lower().split("x")
    else:
        return (832, 480)
    return (int(parts[0]), int(parts[1]))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 TI2V-5B Handler (Text + Image to Video)

    Supports TWO modes:
        - T2V (Text-to-Video): prompt only, no image
        - I2V (Image-to-Video): prompt + image

    Required:
        prompt (str): Text description of the video

    Optional:
        image (str): Base64 encoded input image (if provided, runs I2V mode)
        negative_prompt (str): What to avoid
        size (str): Resolution "WIDTHxHEIGHT" (default: 1280x704 for 720P)
        num_frames (int): Frame count, must be 4n+1 (default: 81)
        sample_steps (int): Denoising steps (default: 50)
        guidance_scale (float): CFG scale (default: 5.0)
        seed (int): Random seed (-1 for random)
        fps (int): Output FPS (default: 16)
    """
    job_input = job.get("input", {})
    image_path = None

    try:
        prompt = job_input.get("prompt", "").strip()
        if not prompt:
            return {"error": "prompt is required"}

        # Image is OPTIONAL - if provided, runs I2V mode; if not, runs T2V mode
        image_data = job_input.get("image", "")
        if image_data:
            image_bytes = decode_base64_image(image_data)
            image_path = save_temp_file(image_bytes, ".png")

        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, low resolution"
        )

        size = job_input.get("size", "1280x704")  # TI2V-5B uses 1280x704 for 720P
        width, height = parse_resolution(size)

        num_frames = job_input.get("num_frames", job_input.get("frame_num", 81))
        num_frames = int(num_frames)
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = max(5, min(257, num_frames))

        sample_steps = int(job_input.get("sample_steps", job_input.get("num_inference_steps", 50)))
        guidance_scale = float(job_input.get("guidance_scale", job_input.get("sample_guide_scale", 5.0)))
        sample_shift = float(job_input.get("sample_shift", job_input.get("flow_shift", 3.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = max(8, min(60, int(job_input.get("fps", 16))))

        logger.info(f"TI2V-5B: {width}x{height}, {num_frames}f, steps={sample_steps}")
        logger.info(f"Prompt: {prompt[:100]}...")

        wan_ti2v = load_model()

        max_area = width * height

        # Load image if provided (I2V mode), otherwise None (T2V mode)
        input_img = None
        if image_path:
            input_img = Image.open(image_path).convert('RGB')
            logger.info("Mode: Image-to-Video (I2V)")
        else:
            logger.info("Mode: Text-to-Video (T2V)")

        with torch.inference_mode():
            video_tensor = wan_ti2v.generate(
                input_prompt=prompt,
                img=input_img,
                size=(width, height),
                frame_num=num_frames,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guidance_scale,
                seed=seed,
                offload_model=False,  # 80GB VRAM - no offloading needed
            )

        from wan.utils.utils import save_video
        output_path = tempfile.mktemp(suffix=".mp4")
        save_video(video_tensor, output_path, fps=fps)

        video_base64 = encode_video_base64(output_path)
        os.remove(output_path)
        if image_path:
            os.remove(image_path)
        cleanup()

        return {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "model": "Wan2.2-TI2V-5B",
            "mode": "I2V" if input_img else "T2V"
        }

    except torch.cuda.OutOfMemoryError as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()
        return {"error": "Out of GPU memory. Reduce resolution or num_frames.", "details": str(e)}
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("Initializing WAN 2.2 TI2V-5B Handler...")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })
