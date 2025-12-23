"""
WAN 2.2 Text-to-Video (T2V) Handler for RunPod Serverless
Model: Wan2.2-T2V-A14B (14B parameters)
GPU: A100 80GB required
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

import runpod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add WAN 2.2 to path
sys.path.insert(0, "/workspace/Wan2.2")

MODEL = None


def load_model():
    """Load T2V-14B model from network volume."""
    global MODEL
    if MODEL is not None:
        return MODEL

    model_dir = os.environ.get("MODEL_DIR", "/workspace/models")
    ckpt_dir = os.path.join(model_dir, "Wan2.2-T2V-A14B")

    logger.info("=" * 60)
    logger.info("Loading WAN 2.2 T2V-A14B model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS["t2v-A14B"]

        MODEL = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("T2V-A14B model loaded successfully!")
        logger.info("=" * 60)
        return MODEL
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_resolution(size: str) -> tuple:
    if not size:
        return (1280, 720)
    if "*" in size:
        parts = size.split("*")
    elif "x" in size.lower():
        parts = size.lower().split("x")
    else:
        return (1280, 720)
    return (int(parts[0]), int(parts[1]))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 T2V-A14B Handler

    Required:
        prompt (str): Text description of the video

    Optional:
        negative_prompt (str): What to avoid
        size (str): Resolution "WIDTHxHEIGHT" (default: 1280x720)
        num_frames (int): Frame count, must be 4n+1 (default: 81)
        sample_steps (int): Denoising steps (default: 50)
        guidance_scale (float): CFG scale (default: 5.0)
        seed (int): Random seed (-1 for random)
        fps (int): Output FPS (default: 16)
    """
    job_input = job.get("input", {})

    try:
        prompt = job_input.get("prompt", "").strip()
        if not prompt:
            return {"error": "prompt is required"}

        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, low resolution, "
            "ugly, bad anatomy, deformed, disfigured, mutation"
        )

        size = job_input.get("size", "1280x720")
        width, height = parse_resolution(size)

        num_frames = job_input.get("num_frames", job_input.get("frame_num", 81))
        num_frames = int(num_frames)
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = max(5, min(257, num_frames))

        sample_steps = int(job_input.get("sample_steps", job_input.get("num_inference_steps", 50)))
        guidance_scale = float(job_input.get("guidance_scale", job_input.get("sample_guide_scale", 5.0)))
        sample_shift = float(job_input.get("sample_shift", job_input.get("flow_shift", 5.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = max(8, min(60, int(job_input.get("fps", 16))))

        logger.info(f"T2V-14B: {width}x{height}, {num_frames}f, steps={sample_steps}")
        logger.info(f"Prompt: {prompt[:100]}...")

        wan_t2v = load_model()

        max_area = width * height

        with torch.inference_mode():
            video_tensor = wan_t2v.generate(
                input_prompt=prompt,
                size=(width, height),
                frame_num=num_frames,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guidance_scale,
                seed=seed,
                offload_model=True,
            )

        from wan.utils.utils import save_video
        output_path = tempfile.mktemp(suffix=".mp4")
        save_video(video_tensor, output_path, fps=fps)

        video_base64 = encode_video_base64(output_path)
        os.remove(output_path)
        cleanup()

        return {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "model": "Wan2.2-T2V-A14B"
        }

    except torch.cuda.OutOfMemoryError as e:
        cleanup()
        return {"error": "Out of GPU memory. Reduce resolution or num_frames.", "details": str(e)}
    except Exception as e:
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("Initializing WAN 2.2 T2V-A14B Handler...")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })
