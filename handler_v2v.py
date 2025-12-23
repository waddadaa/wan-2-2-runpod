"""
WAN 2.2 Video-to-Video / Animate Handler for RunPod Serverless
Model: Wan2.2-Animate-14B
License: Apache 2.0 (Commercial use allowed)

ALL PARAMETERS EXPOSED
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

sys.path.insert(0, "/app/Wan2.2")

MODEL = None

# Model configuration
MODEL_NAME = "Wan2.2-Animate-14B"
HF_REPO_ID = "Wan-AI/Wan2.2-Animate-14B"
WAN_CONFIG_KEY = "animate-14B"


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
    logger.info("This may take a while (~50GB download)")
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
    """Load Animate model, downloading from HuggingFace if needed."""
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
    logger.info("Loading WAN 2.2 Animate model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[WAN_CONFIG_KEY]

        MODEL = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("Animate model loaded successfully!")
        logger.info("=" * 60)
        return MODEL
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def decode_base64_data(base64_string: str) -> bytes:
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
    WAN 2.2 Video-to-Video / Animate Handler - ALL PARAMETERS

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    video (str):
        Base64 encoded source video.
        Supports MP4, MOV, AVI formats.
        Motion from this video drives the output.

    ═══════════════════════════════════════════════════════════════════════════
    CHARACTER REFERENCE (Optional but recommended)
    ═══════════════════════════════════════════════════════════════════════════

    reference_image (str, optional):
        Base64 encoded reference image for character replacement.
        Character appearance from this image replaces source video character.
        Supports PNG, JPG, WEBP.

    ═══════════════════════════════════════════════════════════════════════════
    GENERATION PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str, default: ""):
        Text description of desired output.
        Example: "A woman with blonde hair dancing gracefully"

    negative_prompt (str, default: "..."):
        What to avoid in generation.

    size (str, default: "1280x720"):
        Output resolution. Format: "WIDTHxHEIGHT"

    num_frames / frame_num (int, default: 81):
        Output frames. Must be 4n+1 format.

    ═══════════════════════════════════════════════════════════════════════════
    SAMPLING PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    sample_steps / num_inference_steps (int, default: 50):
        Denoising steps.

    sample_guide_scale / guidance_scale (float, default: 5.0):
        CFG scale.

    sample_shift / flow_shift (float, default: 5.0):
        Flow matching shift.

    sample_solver (str, default: "unipc"):
        Solver: "unipc" or "dpm++"

    ═══════════════════════════════════════════════════════════════════════════
    ANIMATE-SPECIFIC PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    refert_num (int, default: 1):
        Number of temporal guidance reference frames.
        Options: 1 or 5
        5 gives better temporal consistency but slower.

    replace_flag (bool, default: false):
        Enable character replacement mode.
        When true, reference_image character replaces source video character.

    use_relighting_lora (bool, default: false):
        Enable relighting enhancement LoRA.
        Improves lighting consistency.

    pose_video (str, optional):
        Base64 encoded DW-pose sequence video.
        For explicit pose control.

    ═══════════════════════════════════════════════════════════════════════════
    OUTPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    seed / base_seed (int, default: random):
        Random seed.

    fps (int, default: 16):
        Output FPS.

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "width": 1280,
        "height": 720,
        "num_frames": 81,
        "fps": 16,
        "seed": 12345
    }
    """
    job_input = job.get("input", {})
    video_path = None
    ref_image_path = None
    pose_video_path = None

    try:
        # =====================================================================
        # REQUIRED: VIDEO
        # =====================================================================
        video_data = job_input.get("video", "")
        if not video_data:
            return {"error": "video is required (base64 encoded)"}

        video_bytes = decode_base64_data(video_data)
        video_path = save_temp_file(video_bytes, ".mp4")

        # =====================================================================
        # OPTIONAL: REFERENCE IMAGE
        # =====================================================================
        reference_image_data = job_input.get("reference_image", "")
        if reference_image_data:
            ref_bytes = decode_base64_data(reference_image_data)
            ref_image_path = save_temp_file(ref_bytes, ".png")

        # =====================================================================
        # OPTIONAL: POSE VIDEO
        # =====================================================================
        pose_video_data = job_input.get("pose_video", "")
        if pose_video_data:
            pose_bytes = decode_base64_data(pose_video_data)
            pose_video_path = save_temp_file(pose_bytes, ".mp4")

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, low resolution, "
            "ugly, bad anatomy, deformed, static, frozen"
        )

        size = job_input.get("size", "1280x720")
        width, height = parse_resolution(size)

        num_frames = job_input.get("num_frames", job_input.get("frame_num", 81))
        num_frames = int(num_frames)
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = max(5, min(257, num_frames))

        # =====================================================================
        # SAMPLING PARAMETERS
        # =====================================================================
        sample_steps = int(job_input.get("sample_steps", job_input.get("num_inference_steps", 50)))
        sample_guide_scale = float(job_input.get("sample_guide_scale", job_input.get("guidance_scale", 5.0)))
        sample_shift = float(job_input.get("sample_shift", job_input.get("flow_shift", 5.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        # =====================================================================
        # ANIMATE-SPECIFIC PARAMETERS
        # =====================================================================
        refert_num = int(job_input.get("refert_num", 1))
        if refert_num not in [1, 5]:
            refert_num = 1

        replace_flag = job_input.get("replace_flag", False)
        use_relighting_lora = job_input.get("use_relighting_lora", False)

        # =====================================================================
        # OUTPUT PARAMETERS
        # =====================================================================
        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = max(8, min(60, int(job_input.get("fps", 16))))

        # =====================================================================
        # GENERATE
        # =====================================================================
        generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"V2V/Animate: {width}x{height}, {num_frames}f")
        logger.info(f"refert_num={refert_num}, replace={replace_flag}, relighting={use_relighting_lora}")

        pipeline = load_model()

        kwargs = {
            "video": video_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": sample_guide_scale,
            "num_inference_steps": sample_steps,
            "generator": generator,
            "flow_shift": sample_shift,
            "refert_num": refert_num,
            "replace_flag": replace_flag,
            "use_relighting_lora": use_relighting_lora
        }

        if ref_image_path:
            kwargs["reference_image"] = ref_image_path
        if pose_video_path:
            kwargs["pose_video"] = pose_video_path

        with torch.inference_mode():
            output = pipeline(**kwargs)

        output_path = tempfile.mktemp(suffix=".mp4")
        output.save(output_path, fps=fps)
        video_base64 = encode_video_base64(output_path)

        # Cleanup
        os.remove(output_path)
        os.remove(video_path)
        video_path = None
        if ref_image_path:
            os.remove(ref_image_path)
            ref_image_path = None
        if pose_video_path:
            os.remove(pose_video_path)
            pose_video_path = None
        cleanup()

        return {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed
        }

    except torch.cuda.OutOfMemoryError as e:
        for p in [video_path, ref_image_path, pose_video_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        for p in [video_path, ref_image_path, pose_video_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("Initializing WAN 2.2 V2V/Animate Handler...")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })
