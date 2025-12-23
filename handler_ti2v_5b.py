"""
WAN 2.2 Text+Image-to-Video (TI2V) Handler for RunPod Serverless
Model: Wan2.2-TI2V-5B (5B parameters)
GPU: RTX 4090 24GB or A100
License: Apache 2.0 (Commercial use allowed)

Supports TWO modes:
  - T2V (Text-to-Video): prompt only, no image
  - I2V (Image-to-Video): prompt + image
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
        return (1280, 704)
    if "*" in size:
        parts = size.split("*")
    elif "x" in size.lower():
        parts = size.lower().split("x")
    else:
        return (1280, 704)
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

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str):
        Text description of the video to generate.

    ═══════════════════════════════════════════════════════════════════════════
    OPTIONAL - MODE SELECTION
    ═══════════════════════════════════════════════════════════════════════════

    image (str, optional):
        Base64 encoded input image.
        If provided: runs I2V mode (image-to-video)
        If not provided: runs T2V mode (text-to-video)

    ═══════════════════════════════════════════════════════════════════════════
    OPTIONAL - GENERATION PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    negative_prompt (str, default: from config):
        What to avoid in generation.

    size (str, default: "1280x704"):
        Resolution "WIDTHxHEIGHT".
        720P = 1280x704

    frame_num (int, default: 121):
        Frame count. Must be 4n+1 format.
        Max 257 frames.

    ═══════════════════════════════════════════════════════════════════════════
    OPTIONAL - SAMPLING PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    sampling_steps (int, default: T2V=50, I2V=40):
        Diffusion denoising steps.
        Note: I2V mode uses 40 steps by default (faster).

    guide_scale (float, default: 5.0):
        Classifier-free guidance scale.

    shift (float, default: 5.0):
        Flow matching shift parameter.
        Use 3.0 for 480p resolution.

    sample_solver (str, default: "unipc"):
        Solver: "unipc" or "dpm++"

    ═══════════════════════════════════════════════════════════════════════════
    OPTIONAL - OUTPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    seed (int, default: random):
        Random seed for reproducibility. -1 for random.

    fps (int, default: 24):
        Output video FPS.

    offload_model (bool, default: true):
        Offload model to CPU after generation to save VRAM.
        Set to false for faster generation on high-VRAM GPUs (A100 80GB).

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "width": 1280,
        "height": 704,
        "frame_num": 121,
        "fps": 24,
        "seed": 12345,
        "mode": "T2V" or "I2V"
    }
    """
    job_input = job.get("input", {})
    image_path = None

    try:
        # =====================================================================
        # REQUIRED: PROMPT
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        if not prompt:
            return {"error": "prompt is required"}

        # =====================================================================
        # OPTIONAL: IMAGE (determines T2V vs I2V mode)
        # =====================================================================
        image_data = job_input.get("image", "")
        input_img = None
        is_i2v_mode = False

        if image_data:
            image_bytes = decode_base64_image(image_data)
            image_path = save_temp_file(image_bytes, ".png")
            input_img = Image.open(image_path).convert('RGB')
            is_i2v_mode = True

        # =====================================================================
        # NEGATIVE PROMPT
        # =====================================================================
        negative_prompt = job_input.get("negative_prompt", job_input.get("n_prompt", ""))

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        size = job_input.get("size", "1280x704")
        width, height = parse_resolution(size)

        frame_num = job_input.get("frame_num", job_input.get("num_frames", 121))
        frame_num = int(frame_num)
        # Ensure 4n+1 format
        if (frame_num - 1) % 4 != 0:
            frame_num = ((frame_num - 1) // 4) * 4 + 1
        frame_num = max(5, min(257, frame_num))

        # =====================================================================
        # SAMPLING PARAMETERS (with mode-specific defaults)
        # =====================================================================
        # I2V uses 40 steps by default, T2V uses 50
        default_steps = 40 if is_i2v_mode else 50
        sampling_steps = int(job_input.get("sampling_steps", job_input.get("sample_steps", default_steps)))

        guide_scale = float(job_input.get("guide_scale", job_input.get("sample_guide_scale", 5.0)))
        shift = float(job_input.get("shift", job_input.get("sample_shift", 5.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        # =====================================================================
        # OUTPUT PARAMETERS
        # =====================================================================
        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = int(job_input.get("fps", 24))
        fps = max(8, min(60, fps))

        offload_model = job_input.get("offload_model", True)
        if isinstance(offload_model, str):
            offload_model = offload_model.lower() in ("true", "1", "yes")

        # =====================================================================
        # GENERATE
        # =====================================================================
        mode = "I2V" if is_i2v_mode else "T2V"
        logger.info("=" * 60)
        logger.info(f"TI2V-5B Generation - Mode: {mode}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Frames: {frame_num}")
        logger.info(f"  Steps: {sampling_steps}, Guide: {guide_scale}, Shift: {shift}")
        logger.info(f"  Solver: {sample_solver}")
        logger.info(f"  Seed: {seed}, FPS: {fps}")
        logger.info(f"  Offload: {offload_model}")
        logger.info(f"  Prompt: {prompt[:80]}...")
        logger.info("=" * 60)

        wan_ti2v = load_model()

        with torch.inference_mode():
            video_tensor = wan_ti2v.generate(
                input_prompt=prompt,
                img=input_img,
                size=(width, height),
                max_area=width * height,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=offload_model,
            )

        # =====================================================================
        # SAVE VIDEO
        # =====================================================================
        logger.info("Saving output video...")

        from wan.utils.utils import save_video

        output_path = tempfile.mktemp(suffix=".mp4")
        save_video(
            tensor=video_tensor[None],  # Add batch dimension
            save_file=output_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        video_base64 = encode_video_base64(output_path)

        # Cleanup
        os.remove(output_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()

        logger.info(f"TI2V-5B {mode} generation completed successfully!")

        return {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "frame_num": frame_num,
            "fps": fps,
            "seed": seed,
            "mode": mode
        }

    except torch.cuda.OutOfMemoryError as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()
        return {"error": "Out of GPU memory. Reduce resolution or frame_num.", "details": str(e)}
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Initializing WAN 2.2 TI2V-5B Handler...")
    logger.info("Modes: T2V (text-only) and I2V (text+image)")
    logger.info("=" * 60)

    try:
        load_model()
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })
