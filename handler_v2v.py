"""
WAN 2.2 Animate Handler for RunPod Serverless
Model: Wan2.2-Animate-14B
Mode: Animation (motion transfer from driving video to reference image)
License: Apache 2.0 (Commercial use allowed)

INPUT:
  - reference_image: Base64 encoded image (the person/character appearance)
  - driving_video: Base64 encoded video (the motion to transfer)

OUTPUT:
  - Video of reference_image animated with driving_video's motion
"""

import os
import sys
import gc
import shutil
import torch
import base64
import tempfile
import traceback
import logging
from typing import Dict, Any

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
sys.path.insert(0, "/app/Wan2.2/wan/modules/animate/preprocess")

MODEL = None
PREPROCESS_PIPELINE = None

# Model configuration
MODEL_NAME = "Wan2.2-Animate-14B"
HF_REPO_ID = "Wan-AI/Wan2.2-Animate-14B"
WAN_CONFIG_KEY = "animate-14B"

# Preprocessing model URLs (HuggingFace)
PREPROCESS_MODELS = {
    "det": {
        "filename": "yolov10m.onnx",
        "url": "https://huggingface.co/onnx-community/yolov10m/resolve/main/onnx/model.onnx",
        "subdir": "det"
    },
    "pose2d": {
        "filename": "vitpose_h_wholebody.onnx",
        "url": "https://huggingface.co/wanghaofan/Sonic/resolve/main/vitpose-h-wholebody.onnx",
        "subdir": "pose2d"
    }
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file from URL to destination path."""
    import urllib.request
    try:
        logger.info(f"Downloading {url} to {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded successfully: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def ensure_preprocess_models(preprocess_dir: str) -> bool:
    """Ensure preprocessing models are downloaded."""
    all_present = True

    for model_key, model_info in PREPROCESS_MODELS.items():
        model_path = os.path.join(preprocess_dir, model_info["subdir"], model_info["filename"])

        if os.path.exists(model_path):
            logger.info(f"Preprocess model found: {model_path}")
        else:
            logger.info(f"Preprocess model not found: {model_path}")
            if not download_file(model_info["url"], model_path):
                all_present = False

    return all_present


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


def load_preprocess_pipeline():
    """Load the preprocessing pipeline for pose extraction."""
    global PREPROCESS_PIPELINE
    if PREPROCESS_PIPELINE is not None:
        return PREPROCESS_PIPELINE

    preprocess_dir = os.environ.get("PREPROCESS_MODEL_DIR", "/runpod-volume/models/preprocess")

    # Ensure models are downloaded
    if not ensure_preprocess_models(preprocess_dir):
        raise RuntimeError("Failed to download preprocessing models")

    det_path = os.path.join(preprocess_dir, "det", "yolov10m.onnx")
    pose2d_path = os.path.join(preprocess_dir, "pose2d", "vitpose_h_wholebody.onnx")

    logger.info("Loading preprocessing pipeline...")
    logger.info(f"  Detection model: {det_path}")
    logger.info(f"  Pose model: {pose2d_path}")

    try:
        from process_pipepline import ProcessPipeline
        PREPROCESS_PIPELINE = ProcessPipeline(
            det_checkpoint_path=det_path,
            pose2d_checkpoint_path=pose2d_path,
            sam_checkpoint_path=None,  # Not needed for animation mode
            flux_kontext_path=None     # Not needed for basic animation
        )
        logger.info("Preprocessing pipeline loaded successfully!")
        return PREPROCESS_PIPELINE
    except Exception as e:
        logger.error(f"Failed to load preprocessing pipeline: {e}")
        raise


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
            use_relighting_lora=True,  # Always enabled for better lighting quality
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


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_preprocessing(video_path: str, reference_image_path: str, output_dir: str,
                      resolution: tuple = (1280, 720), fps: int = 30,
                      retarget_flag: bool = False) -> bool:
    """
    Run preprocessing to extract pose and face data from driving video.

    Creates:
      - src_pose.mp4: Skeleton pose video
      - src_face.mp4: Cropped face video (512x512)
      - src_ref.png: Reference image (copied)

    Args:
        retarget_flag: Enable pose retargeting to handle different body proportions
                       between reference image and driving video. Works best when both
                       characters are in front-facing poses.
    """
    pipeline = load_preprocess_pipeline()

    logger.info(f"Running preprocessing...")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Reference: {reference_image_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Retarget: {retarget_flag}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        result = pipeline(
            video_path=video_path,
            refer_image_path=reference_image_path,
            output_path=output_dir,
            resolution_area=list(resolution),
            fps=fps,
            replace_flag=False,  # Animation mode, not replacement
            retarget_flag=retarget_flag,  # Pose retargeting for different body proportions
            use_flux=False       # No FLUX editing
        )

        # Verify output files exist
        required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
        for f in required_files:
            fpath = os.path.join(output_dir, f)
            if not os.path.exists(fpath):
                raise RuntimeError(f"Preprocessing failed to create: {f}")

        logger.info("Preprocessing completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 Animate Handler - Animation Mode

    Transfers motion from a driving video to a reference image.

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    reference_image (str):
        Base64 encoded reference image.
        This is the person/character whose appearance will be used.
        Supports PNG, JPG, WEBP.

    driving_video (str):
        Base64 encoded driving video.
        Motion from this video will be transferred to the reference image.
        Supports MP4, MOV, AVI.

    ═══════════════════════════════════════════════════════════════════════════
    OPTIONAL PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str, default: ""):
        Text description (optional, minimal effect).

    resolution (list, default: [1280, 720]):
        Output resolution [width, height].

    fps (int, default: 30):
        Frames per second for processing and output.

    clip_len (int, default: 77):
        Frames per generation clip. Must be 4n+1 format.
        Higher = more temporal consistency, more VRAM.

    refert_num (int, default: 1):
        Temporal guidance frames. Options: 1 or 5.
        5 = better consistency, slower.

    sampling_steps (int, default: 20):
        Diffusion denoising steps. Higher = better quality, slower.

    guide_scale (float, default: 1.0):
        CFG scale. Usually keep at 1.0 for animate.

    shift (float, default: 5.0):
        Flow matching shift parameter.

    sample_solver (str, default: "dpm++"):
        Solver: "dpm++" or "unipc"

    seed (int, default: random):
        Random seed for reproducibility.

    offload_model (bool, default: True):
        Offload model to CPU after forward pass to save VRAM.

    retarget_flag (bool, default: False):
        Enable pose retargeting to handle different body proportions
        between reference image and driving video.
        Recommended when character sizes/proportions differ significantly.
        Works best when both characters are in front-facing poses.

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "resolution": [1280, 720],
        "fps": 30,
        "seed": 12345
    }
    """
    job_input = job.get("input", {})
    temp_dir = None
    video_path = None
    ref_image_path = None

    try:
        # =====================================================================
        # REQUIRED: REFERENCE IMAGE
        # =====================================================================
        reference_image_data = job_input.get("reference_image", "")
        if not reference_image_data:
            return {"error": "reference_image is required (base64 encoded)"}

        ref_bytes = decode_base64_data(reference_image_data)
        ref_image_path = save_temp_file(ref_bytes, ".png")

        # =====================================================================
        # REQUIRED: DRIVING VIDEO
        # =====================================================================
        driving_video_data = job_input.get("driving_video", "")
        if not driving_video_data:
            return {"error": "driving_video is required (base64 encoded)"}

        video_bytes = decode_base64_data(driving_video_data)
        video_path = save_temp_file(video_bytes, ".mp4")

        # =====================================================================
        # OPTIONAL PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        if not prompt:
            prompt = "视频中的人在做动作"  # Default: "Person doing actions in video"

        negative_prompt = job_input.get("negative_prompt", "")

        resolution = job_input.get("resolution", [1280, 720])
        if isinstance(resolution, str):
            if "x" in resolution.lower():
                parts = resolution.lower().split("x")
                resolution = [int(parts[0]), int(parts[1])]
            elif "*" in resolution:
                parts = resolution.split("*")
                resolution = [int(parts[0]), int(parts[1])]
        resolution = tuple(resolution)

        fps = int(job_input.get("fps", 30))
        clip_len = int(job_input.get("clip_len", job_input.get("frame_num", 77)))

        # Ensure clip_len is 4n+1 format
        if (clip_len - 1) % 4 != 0:
            clip_len = ((clip_len - 1) // 4) * 4 + 1

        refert_num = int(job_input.get("refert_num", 1))
        if refert_num not in [1, 5]:
            refert_num = 1

        sampling_steps = int(job_input.get("sampling_steps", job_input.get("sample_steps", 20)))
        guide_scale = float(job_input.get("guide_scale", job_input.get("sample_guide_scale", 1.0)))
        shift = float(job_input.get("shift", job_input.get("sample_shift", 5.0)))
        sample_solver = job_input.get("sample_solver", "dpm++")

        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        offload_model = job_input.get("offload_model", True)

        retarget_flag = job_input.get("retarget_flag", False)
        if isinstance(retarget_flag, str):
            retarget_flag = retarget_flag.lower() in ("true", "1", "yes")

        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        temp_dir = tempfile.mkdtemp(prefix="wan_animate_")

        logger.info("=" * 60)
        logger.info("STEP 1: Preprocessing")
        logger.info("=" * 60)

        run_preprocessing(
            video_path=video_path,
            reference_image_path=ref_image_path,
            output_dir=temp_dir,
            resolution=resolution,
            fps=fps,
            retarget_flag=retarget_flag
        )

        # =====================================================================
        # STEP 2: GENERATION
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Generation")
        logger.info(f"  clip_len={clip_len}, refert_num={refert_num}")
        logger.info(f"  sampling_steps={sampling_steps}, guide_scale={guide_scale}")
        logger.info(f"  shift={shift}, solver={sample_solver}")
        logger.info(f"  seed={seed}")
        logger.info("=" * 60)

        pipeline = load_model()

        with torch.inference_mode():
            video_tensor = pipeline.generate(
                src_root_path=temp_dir,
                replace_flag=False,
                clip_len=clip_len,
                refert_num=refert_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                input_prompt=prompt,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=offload_model
            )

        # =====================================================================
        # STEP 3: SAVE OUTPUT
        # =====================================================================
        logger.info("Saving output video...")

        from wan.utils.utils import save_video

        output_path = os.path.join(temp_dir, "output.mp4")
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
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if ref_image_path and os.path.exists(ref_image_path):
            os.remove(ref_image_path)
        cleanup()

        logger.info("Generation completed successfully!")

        return {
            "video": video_base64,
            "format": "mp4",
            "resolution": list(resolution),
            "fps": fps,
            "seed": seed
        }

    except torch.cuda.OutOfMemoryError as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        for p in [video_path, ref_image_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        for p in [video_path, ref_image_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Initializing WAN 2.2 Animate Handler...")
    logger.info("Mode: Animation (motion transfer)")
    logger.info("=" * 60)

    try:
        # Pre-load preprocessing pipeline
        load_preprocess_pipeline()
        # Pre-load main model
        load_model()
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })
