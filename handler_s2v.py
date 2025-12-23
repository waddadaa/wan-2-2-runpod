"""
WAN 2.2 Speech-to-Video (S2V) Handler for RunPod Serverless
Model: Wan2.2-S2V-14B
License: Apache 2.0 (Commercial use allowed)

Generates talking head videos with lip-sync from audio or text-to-speech.

INPUT:
  - image: Base64 encoded reference face image
  - audio: Base64 encoded audio file (or use TTS mode)

OUTPUT:
  - Video with lip-synced speech
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
MODEL_NAME = "Wan2.2-S2V-14B"
HF_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
WAN_CONFIG_KEY = "s2v-14B"


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
    logger.info("This may take a while (~43GB download)")
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
    """Load the S2V model, downloading from HuggingFace if needed."""
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
    logger.info("Loading WAN 2.2 S2V model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[WAN_CONFIG_KEY]

        MODEL = wan.WanS2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("S2V model loaded successfully!")
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


def get_audio_duration(audio_path: str) -> float:
    try:
        import soundfile as sf
        data, samplerate = sf.read(audio_path)
        return len(data) / samplerate
    except:
        return 5.0


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 Speech-to-Video Handler

    Generates talking head videos with lip-sync from audio or text-to-speech.

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    image (str):
        Base64 encoded reference face image.
        Clear frontal face recommended for best results.
        Supports PNG, JPG, WEBP.

    ═══════════════════════════════════════════════════════════════════════════
    AUDIO INPUT (Option A: Pre-recorded Audio)
    ═══════════════════════════════════════════════════════════════════════════

    audio (str):
        Base64 encoded audio file.
        Supports WAV, MP3, FLAC, OGG.
        Speech audio drives lip sync.

    ═══════════════════════════════════════════════════════════════════════════
    AUDIO INPUT (Option B: Text-to-Speech)
    ═══════════════════════════════════════════════════════════════════════════

    enable_tts (bool, default: false):
        Enable text-to-speech synthesis using CosyVoice2.
        When true, generates audio from text instead of using audio input.
        Note: CosyVoice2 will be downloaded on first use (~1GB).

    tts_text (str, required if enable_tts=true):
        Text to synthesize into speech.

    tts_prompt_audio (str, optional):
        Base64 encoded reference voice sample for voice cloning.
        Model will attempt to clone this voice.

    tts_prompt_text (str, optional):
        Transcription of tts_prompt_audio.
        Helps model understand reference voice better.

    ═══════════════════════════════════════════════════════════════════════════
    GENERATION PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str, default: ""):
        Additional text guidance for video style.
        Example: "Professional news anchor in studio"

    negative_prompt (str, default: from config):
        What to avoid in generation.

    max_area (int, default: 921600):
        Maximum pixel area (width * height).
        Default is 1280x720 = 921600.
        Model auto-calculates resolution from reference image.

    ═══════════════════════════════════════════════════════════════════════════
    SAMPLING PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    sampling_steps (int, default: 40):
        Diffusion denoising steps. Higher = better quality, slower.

    guide_scale (float, default: 4.5):
        Classifier-free guidance scale.

    shift (float, default: 3.0):
        Flow matching shift parameter.
        Note: Use 3.0 for 480p, can increase for higher res.

    sample_solver (str, default: "unipc"):
        Solver: "unipc" or "dpm++"

    ═══════════════════════════════════════════════════════════════════════════
    S2V-SPECIFIC PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    num_repeat (int, default: auto):
        Number of video clips to generate.
        Auto-calculated from audio duration if not specified.
        For long audio, model generates multiple clips and concatenates.

    infer_frames (int, default: 80):
        Frames per clip. Must be divisible by 4.
        Higher = longer clips but more VRAM.

    init_first_frame (bool, default: false):
        Use reference image as the exact first frame.
        When false, model may slightly modify the appearance.

    pose_video (str, optional):
        Base64 encoded pose video for pose-driven generation.
        If provided, uses pose sequence to drive the generated video.

    ═══════════════════════════════════════════════════════════════════════════
    OUTPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    seed (int, default: random):
        Random seed for reproducibility. -1 for random.

    fps (int, default: 16):
        Output video FPS.
        Note: Model internally uses 16 FPS for audio sync calculations.

    offload_model (bool, default: true):
        Offload model to CPU after generation to save VRAM.

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "fps": 16,
        "seed": 12345,
        "audio_duration": 5.0,
        "tts_used": false
    }
    """
    job_input = job.get("input", {})
    audio_path = None
    image_path = None
    tts_prompt_audio_path = None
    pose_video_path = None
    output_path = None

    try:
        # =====================================================================
        # REQUIRED: IMAGE
        # =====================================================================
        image_data = job_input.get("image", "")
        if not image_data:
            return {"error": "image is required (base64 encoded reference face image)"}

        image_bytes = decode_base64_data(image_data)
        image_path = save_temp_file(image_bytes, ".png")

        # =====================================================================
        # AUDIO INPUT: Either direct audio or TTS
        # =====================================================================
        enable_tts = job_input.get("enable_tts", False)
        if isinstance(enable_tts, str):
            enable_tts = enable_tts.lower() in ("true", "1", "yes")

        tts_text = None
        tts_prompt_audio_for_gen = None
        tts_prompt_text_for_gen = None

        if enable_tts:
            # TTS Mode - let the model handle TTS internally
            tts_text = job_input.get("tts_text", "")
            if not tts_text:
                return {"error": "tts_text is required when enable_tts=true"}

            tts_prompt_audio_data = job_input.get("tts_prompt_audio", "")
            tts_prompt_text_for_gen = job_input.get("tts_prompt_text", None)

            if tts_prompt_audio_data:
                tts_prompt_audio_bytes = decode_base64_data(tts_prompt_audio_data)
                tts_prompt_audio_path = save_temp_file(tts_prompt_audio_bytes, ".wav")
                tts_prompt_audio_for_gen = tts_prompt_audio_path

            # Audio path will be generated by the model's TTS
            audio_path = None
            audio_duration = len(tts_text) * 0.1  # Rough estimate for logging
            logger.info(f"TTS mode enabled: '{tts_text[:50]}...'")
        else:
            # Direct audio mode
            audio_data = job_input.get("audio", "")
            if not audio_data:
                return {"error": "audio is required (base64 encoded) or set enable_tts=true"}

            audio_bytes = decode_base64_data(audio_data)
            if audio_bytes[:4] == b'RIFF':
                audio_suffix = ".wav"
            elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                audio_suffix = ".mp3"
            else:
                audio_suffix = ".wav"
            audio_path = save_temp_file(audio_bytes, audio_suffix)
            audio_duration = get_audio_duration(audio_path)

        logger.info(f"Audio duration: {audio_duration:.2f}s")

        # =====================================================================
        # OPTIONAL: POSE VIDEO
        # =====================================================================
        pose_video_data = job_input.get("pose_video", "")
        if pose_video_data:
            pose_video_bytes = decode_base64_data(pose_video_data)
            pose_video_path = save_temp_file(pose_video_bytes, ".mp4")
            logger.info("Pose video provided for pose-driven generation")

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        negative_prompt = job_input.get("negative_prompt", "")

        # Max area for resolution calculation
        max_area = int(job_input.get("max_area", 1280 * 720))

        # =====================================================================
        # SAMPLING PARAMETERS (with correct defaults from official config)
        # =====================================================================
        sampling_steps = int(job_input.get("sampling_steps", job_input.get("sample_steps", 40)))
        guide_scale = float(job_input.get("guide_scale", job_input.get("sample_guide_scale", 4.5)))
        shift = float(job_input.get("shift", job_input.get("sample_shift", 3.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        # =====================================================================
        # S2V-SPECIFIC PARAMETERS
        # =====================================================================
        num_repeat = job_input.get("num_repeat", job_input.get("num_clip", None))
        if num_repeat is not None:
            num_repeat = int(num_repeat)

        infer_frames = int(job_input.get("infer_frames", 80))
        # Ensure divisible by 4
        if infer_frames % 4 != 0:
            infer_frames = (infer_frames // 4) * 4
        infer_frames = max(4, infer_frames)

        init_first_frame = job_input.get("init_first_frame", job_input.get("start_from_ref", False))
        if isinstance(init_first_frame, str):
            init_first_frame = init_first_frame.lower() in ("true", "1", "yes")

        # =====================================================================
        # OUTPUT PARAMETERS
        # =====================================================================
        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = int(job_input.get("fps", 16))
        offload_model = job_input.get("offload_model", True)

        # =====================================================================
        # GENERATE
        # =====================================================================
        logger.info("=" * 60)
        logger.info("Starting S2V Generation")
        logger.info(f"  max_area={max_area}")
        logger.info(f"  infer_frames={infer_frames}, num_repeat={num_repeat}")
        logger.info(f"  sampling_steps={sampling_steps}, guide_scale={guide_scale}")
        logger.info(f"  shift={shift}, solver={sample_solver}")
        logger.info(f"  init_first_frame={init_first_frame}")
        logger.info(f"  seed={seed}, fps={fps}")
        logger.info(f"  TTS enabled={enable_tts}")
        logger.info("=" * 60)

        wan_s2v = load_model()

        with torch.inference_mode():
            video_tensor = wan_s2v.generate(
                input_prompt=prompt,
                ref_image_path=image_path,
                audio_path=audio_path,
                enable_tts=enable_tts,
                tts_prompt_audio=tts_prompt_audio_for_gen,
                tts_prompt_text=tts_prompt_text_for_gen,
                tts_text=tts_text,
                num_repeat=num_repeat,
                pose_video=pose_video_path,
                max_area=max_area,
                infer_frames=infer_frames,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=offload_model,
                init_first_frame=init_first_frame,
            )

        # =====================================================================
        # SAVE VIDEO
        # =====================================================================
        logger.info("Saving output video...")

        from wan.utils.utils import save_video, merge_video_audio

        output_path = tempfile.mktemp(suffix=".mp4")

        # Step 1: Save video without audio
        save_video(
            tensor=video_tensor[None],  # Add batch dimension
            save_file=output_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        # Step 2: Merge audio into video
        # Determine audio path (either user-provided or TTS-generated)
        if enable_tts:
            # TTS generates audio to 'tts.wav' in current directory
            tts_audio_path = "tts.wav"
            if os.path.exists(tts_audio_path):
                merge_video_audio(video_path=output_path, audio_path=tts_audio_path)
                # Clean up TTS audio
                os.remove(tts_audio_path)
        elif audio_path:
            merge_video_audio(video_path=output_path, audio_path=audio_path)

        video_base64 = encode_video_base64(output_path)

        # Get actual audio duration after potential TTS
        if enable_tts and os.path.exists("tts.wav"):
            audio_duration = get_audio_duration("tts.wav")

        # Cleanup
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        if tts_prompt_audio_path and os.path.exists(tts_prompt_audio_path):
            os.remove(tts_prompt_audio_path)
        if pose_video_path and os.path.exists(pose_video_path):
            os.remove(pose_video_path)
        cleanup()

        logger.info("S2V generation completed successfully!")

        return {
            "video": video_base64,
            "format": "mp4",
            "fps": fps,
            "seed": seed,
            "audio_duration": round(audio_duration, 2),
            "tts_used": enable_tts
        }

    except torch.cuda.OutOfMemoryError as e:
        for p in [audio_path, image_path, tts_prompt_audio_path, pose_video_path, output_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        for p in [audio_path, image_path, tts_prompt_audio_path, pose_video_path, output_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Initializing WAN 2.2 S2V Handler...")
    logger.info("Mode: Speech-to-Video (talking head generation)")
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
