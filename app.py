# Complete Gradio Application for Wan 2.1 Video Generation
# Run this in Google Colab
import subprocess
import sys
import os
from pathlib import Path
# Install Gradio first if not installed
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    import gradio as gr
# Global state management
class AppState:
    def __init__(self):
        self.env_setup_complete = False
        self.models_downloaded = False
        self.models_loaded = False
        self.comfy_path = os.path.join(os.getcwd(), "ComfyUI")
        self.output_dir = os.path.join(self.comfy_path, "output")
        self.useQ6 = False
        # Model components
        self.unet_loader = None
        self.clip_loader = None
        self.clip_encode_positive = None
        self.clip_encode_negative = None
        self.vae_loader = None
        self.empty_latent_video = None
        self.ksampler = None
        self.vae_decode = None
state = AppState()
# ==================== ENVIRONMENT SETUP ====================
def install_dependencies(progress=gr.Progress()):
    """Step 1: Install all required dependencies"""
    logs = []
    try:
        progress(0.1, desc="Installing PyTorch...")
        logs.append("📦 Installing PyTorch 2.6.0...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.6.0", "torchvision==0.21.0", "--force-reinstall"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logs.append(f"Warning: {result.stderr}")
        logs.append("✅ PyTorch installed")
        progress(0.3, desc="Installing core packages...")
        logs.append("📦 Installing torchsde, einops, diffusers...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torchsde", "einops", "diffusers", "accelerate"],
                        check=True, capture_output=True)
        logs.append("✅ Core packages installed")
        progress(0.4, desc="Installing xformers...")
        logs.append("📦 Installing xformers...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "xformers==0.0.29.post2"],
                        check=True, capture_output=True)
        logs.append("✅ Xformers installed")
        progress(0.5, desc="Installing media packages...")
        logs.append("📦 Installing av and imageio...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "av", "imageio", "imageio-ffmpeg"],
                        check=True, capture_output=True)
        logs.append("✅ Media packages installed")
        progress(0.6, desc="Cloning ComfyUI...")
        if not os.path.exists(state.comfy_path):
            logs.append("📦 Cloning ComfyUI repository...")
            subprocess.run(["git", "clone", "https://github.com/Isi-dev/ComfyUI",
                          state.comfy_path], check=True, capture_output=True)
            logs.append("✅ ComfyUI cloned")
        else:
            logs.append("✅ ComfyUI already exists")
        progress(0.7, desc="Setting up custom nodes...")
        custom_nodes_path = os.path.join(state.comfy_path, "custom_nodes")
        os.makedirs(custom_nodes_path, exist_ok=True)
        gguf_path = os.path.join(custom_nodes_path, "ComfyUI_GGUF")
        if not os.path.exists(gguf_path):
            logs.append("📦 Cloning ComfyUI_GGUF...")
            subprocess.run(["git", "clone",
                            "https://github.com/Isi-dev/ComfyUI_GGUF.git",
                            gguf_path], check=True, capture_output=True)
            logs.append("📦 Installing GGUF requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                            os.path.join(gguf_path, "requirements.txt")],
                            check=True, capture_output=True)
            logs.append("✅ ComfyUI_GGUF installed")
        else:
            logs.append("✅ ComfyUI_GGUF already exists")
        progress(0.9, desc="Installing system packages...")
        logs.append("📦 Installing aria2 and ffmpeg...")
        subprocess.run(["apt", "-y", "install", "-qq", "aria2", "ffmpeg"],
                        check=True, capture_output=True)
        logs.append("✅ System packages installed")
        progress(1.0, desc="Complete!")
        state.env_setup_complete = True
        logs.append("✅ Environment setup complete!")
        logs.append("⚠️  Please restart the runtime (Runtime > Restart runtime) before proceeding to avoid import conflicts!")
        return "\n".join(logs), gr.update(interactive=True)
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")
        return "\n".join(logs), gr.update(interactive=False)
def download_models(use_q6, progress=gr.Progress()):
    """Step 2: Download model files"""
    logs = []
    state.useQ6 = use_q6
    try:
        os.makedirs(os.path.join(state.comfy_path, "models/unet"), exist_ok=True)
        os.makedirs(os.path.join(state.comfy_path, "models/text_encoders"), exist_ok=True)
        os.makedirs(os.path.join(state.comfy_path, "models/vae"), exist_ok=True)
        progress(0.1, desc="Downloading UNET model...")
        if use_q6:
            model_name = "wan2.1-t2v-14b-Q6_K.gguf"
            model_url = "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q6_K.gguf"
        else:
            model_name = "wan2.1-t2v-14b-Q5_0.gguf"
            model_url = "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q5_0.gguf"
        model_path = os.path.join(state.comfy_path, "models/unet", model_name)
        if not os.path.exists(model_path):
            logs.append(f"📥 Downloading {model_name} (this may take 10-15 minutes)...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16",
                            "-s", "16", "-k", "1M", model_url,
                            "-d", os.path.join(state.comfy_path, "models/unet"), "-o", model_name],
                            check=True)
            logs.append(f"✅ {model_name} downloaded")
        else:
            logs.append(f"✅ {model_name} already exists")
        progress(0.4, desc="Downloading text encoder...")
        text_encoder_path = os.path.join(state.comfy_path, "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        if not os.path.exists(text_encoder_path):
            logs.append("📥 Downloading text encoder...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16",
                            "-s", "16", "-k", "1M",
                            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                            "-d", os.path.join(state.comfy_path, "models/text_encoders"),
                            "-o", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
                            check=True)
            logs.append("✅ Text encoder downloaded")
        else:
            logs.append("✅ Text encoder already exists")
        progress(0.7, desc="Downloading VAE...")
        vae_path = os.path.join(state.comfy_path, "models/vae/wan_2.1_vae.safetensors")
        if not os.path.exists(vae_path):
            logs.append("📥 Downloading VAE...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16",
                            "-s", "16", "-k", "1M",
                            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
                            "-d", os.path.join(state.comfy_path, "models/vae"),
                            "-o", "wan_2.1_vae.safetensors"],
                            check=True)
            logs.append("✅ VAE downloaded")
        else:
            logs.append("✅ VAE already exists")
        progress(1.0, desc="Complete!")
        state.models_downloaded = True
        logs.append("✅ All models downloaded successfully!")
        return "\n".join(logs), gr.update(interactive=True)
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")
        return "\n".join(logs), gr.update(interactive=False)
# ==================== UTILITY FUNCTIONS ====================
def initialize_comfy_nodes(progress=gr.Progress()):
    """Step 3: Initialize ComfyUI nodes - Import after runtime restart"""
    logs = []
    try:
        progress(0.1, desc="Importing torch and numpy...")
        logs.append("📦 Importing core libraries...")
        # Import these AFTER the runtime restart
        import torch
        import numpy as np
        import gc
        import random
        from PIL import Image
        import imageio
        logs.append("✅ Core libraries imported")
        progress(0.2, desc="Adding ComfyUI to path...")
        if state.comfy_path not in sys.path:
            sys.path.insert(0, state.comfy_path)
        logs.append("✅ ComfyUI path configured")
        progress(0.4, desc="Importing ComfyUI modules...")
        logs.append("📦 Importing ComfyUI nodes...")
        from nodes import (
            CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader, KSampler
        )
        from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
        from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
        logs.append("✅ ComfyUI modules imported")
        progress(0.7, desc="Initializing nodes...")
        logs.append("⚙️ Initializing node instances...")
        state.unet_loader = UnetLoaderGGUF()
        state.clip_loader = CLIPLoader()
        state.clip_encode_positive = CLIPTextEncode()
        state.clip_encode_negative = CLIPTextEncode()
        state.vae_loader = VAELoader()
        state.empty_latent_video = EmptyHunyuanLatentVideo()
        state.ksampler = KSampler()
        state.vae_decode = VAEDecode()
        logs.append("✅ All nodes initialized")
        progress(1.0, desc="Complete!")
        state.models_loaded = True
        logs.append("✅ Model loading system ready!")
        return "\n".join(logs), gr.update(interactive=True)
    except Exception as e:
        import traceback
        logs.append(f"❌ Error: {str(e)}")
        logs.append(f"Traceback: {traceback.format_exc()}")
        logs.append("\n⚠️  If you see import errors, please restart the runtime first!")
        return "\n".join(logs), gr.update(interactive=False)
def clear_memory():
    """Clear GPU and CPU memory"""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
def save_as_mp4(images, filename_prefix, fps, output_dir):
    """Save tensor images as MP4 video"""
    import numpy as np
    import imageio
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path
def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=10, output_dir=state.output_dir):
    """Save tensor images as WEBM video"""
    import numpy as np
    import imageio
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webm"
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
    with imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=fps, codec=codec, quality=quality) as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path
def save_as_image(image, filename_prefix, output_dir):
    """Save single tensor image as PNG"""
    import numpy as np
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"
    frame = (image.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(frame).save(output_path)
    return output_path
# ==================== VIDEO GENERATION ====================
def generate_video(
    positive_prompt, negative_prompt, width, height, seed, steps,
    cfg_scale, sampler_name, scheduler, frames, fps, output_format,
    use_random_seed, progress=gr.Progress()
):
    """Step 4: Generate video"""
    if not state.models_loaded:
        return None, "❌ Please complete setup steps first! Make sure to restart runtime after Step 1."
    logs = []
    try:
        import torch
        import gc
        import random
        if use_random_seed:
            seed = random.randint(0, 2**32 - 1)
            logs.append(f"🎲 Random seed: {seed}")
        with torch.inference_mode():
            progress(0.1, desc="Loading text encoder...")
            logs.append("📝 Loading text encoder...")
            clip = state.clip_loader.load_clip(
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"
            )[0]
            progress(0.2, desc="Encoding prompts...")
            logs.append("📝 Encoding prompts...")
            positive = state.clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = state.clip_encode_negative.encode(clip, negative_prompt)[0]
            del clip
            torch.cuda.empty_cache()
            gc.collect()
            progress(0.3, desc="Creating latent space...")
            logs.append("🎨 Creating latent space...")
            empty_latent = state.empty_latent_video.generate(width, height, frames, 1)[0]
            progress(0.4, desc="Loading UNET model...")
            logs.append("🧠 Loading UNET model...")
            if state.useQ6:
                model = state.unet_loader.load_unet("wan2.1-t2v-14b-Q6_K.gguf")[0]
            else:
                model = state.unet_loader.load_unet("wan2.1-t2v-14b-Q5_0.gguf")[0]
            progress(0.5, desc="Generating video (this may take several minutes)...")
            logs.append("🎬 Generating video...")
            sampled = state.ksampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=empty_latent
            )[0]
            del model
            torch.cuda.empty_cache()
            gc.collect()
            progress(0.8, desc="Loading VAE...")
            logs.append("🎨 Loading VAE...")
            vae = state.vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            progress(0.9, desc="Decoding latents...")
            logs.append("🎬 Decoding latents...")
            decoded = state.vae_decode.decode(vae, sampled)[0]
            del vae
            torch.cuda.empty_cache()
            gc.collect()
            progress(0.95, desc="Saving output...")
            output_path = ""
            timestamp = random.randint(1000, 9999)
            if frames == 1:
                logs.append("💾 Saving as image...")
                output_path = save_as_image(
                    decoded[0], f"video_{timestamp}", state.output_dir
                )
            else:
                if output_format.lower() == "webm":
                    logs.append("💾 Saving as WEBM...")
                    output_path = save_as_webm(
                        decoded, f"video_{timestamp}", fps=fps,
                        codec="vp9", quality=10, output_dir=state.output_dir
                    )
                else:
                    logs.append("💾 Saving as MP4...")
                    output_path = save_as_mp4(
                        decoded, f"video_{timestamp}", fps, state.output_dir
                    )
            progress(1.0, desc="Complete!")
            clear_memory()
            logs.append(f"✅ Generation complete! Seed: {seed}")
            logs.append(f"📁 Saved to: {output_path}")
            return output_path, "\n".join(logs)
    except Exception as e:
        import traceback
        logs.append(f"❌ Error: {str(e)}")
        logs.append(f"Traceback: {traceback.format_exc()}")
        clear_memory()
        return None, "\n".join(logs)
# ==================== GRADIO INTERFACE ====================
def create_interface():
    """Create the main Gradio interface"""
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
        title="Wan 2.1 Video Generator",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .step-box {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }
        .warning-box {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
        }
        """
    ) as app:
        gr.Markdown("""
        # 🎬 Wan 2.1 Text-to-Video Generator
        ### Create stunning AI-generated videos with state-of-the-art diffusion models
        **Important:** Follow the steps in order. After Step 1, you MUST restart the runtime before proceeding!
        """)
        with gr.Tabs() as tabs:
            # ========== TAB 1: ENVIRONMENT SETUP ==========
            with gr.Tab("1️⃣ Environment Setup", id=0):
                gr.Markdown("""
                ## 📦 Step 1: Install Dependencies
                Install PyTorch, ComfyUI, and required packages. This takes 5-10 minutes.
                **⚠️ IMPORTANT:** After this step completes, you MUST restart the runtime:
                - Click `Runtime` → `Restart runtime` in the Colab menu
                - Then return here and continue to Step 2
                """)
                setup_btn = gr.Button("🚀 Start Installation", variant="primary", size="lg")
                setup_logs = gr.Textbox(label="Installation Logs", lines=15, interactive=False)
                with gr.Row():
                    with gr.Column():
                        setup_status = gr.Markdown("⏳ Waiting to start...")
                gr.Markdown("""
                <div class="warning-box">
                ⚠️ <strong>After installation completes:</strong><br>
                1. Go to <code>Runtime</code> → <code>Restart runtime</code><br>
                2. Reopen this Gradio interface<br>
                3. Continue to Step 2
                </div>
                """)
                next_to_download = gr.Button("➡️ Next: Download Models", interactive=False, size="lg")
                setup_btn.click(
                    fn=install_dependencies,
                    outputs=[setup_logs, next_to_download]
                ).then(
                    fn=lambda: gr.update(value="✅ Installation complete! **Please restart runtime now!**"),
                    outputs=[setup_status]
                )
                next_to_download.click(
                    fn=lambda: gr.update(selected=1),
                    outputs=[tabs]
                )
            # ========== TAB 2: MODEL DOWNLOAD ==========
            with gr.Tab("2️⃣ Download Models", id=1):
                gr.Markdown("""
                ## 📥 Step 2: Download Model Files
                Download the AI models needed for video generation (~15-20 GB total).
                **Note:** Make sure you've restarted the runtime after Step 1!
                """)
                with gr.Row():
                    use_q6_checkbox = gr.Checkbox(
                        label="Use Q6 Model (Higher quality, requires more VRAM - 24GB+ recommended)",
                        value=False,
                        info="Q5 model works on most GPUs (12GB+), Q6 gives better quality but needs 24GB+"
                    )
                download_btn = gr.Button("📥 Download Models", variant="primary", size="lg")
                download_logs = gr.Textbox(label="Download Logs", lines=15, interactive=False)
                download_status = gr.Markdown("⏳ Waiting to start...")
                next_to_init = gr.Button("➡️ Next: Initialize System", interactive=False, size="lg")
                download_btn.click(
                    fn=download_models,
                    inputs=[use_q6_checkbox],
                    outputs=[download_logs, next_to_init]
                ).then(
                    fn=lambda: gr.update(value="✅ Models downloaded successfully!"),
                    outputs=[download_status]
                )
                next_to_init.click(
                    fn=lambda: gr.update(selected=2),
                    outputs=[tabs]
                )
            # ========== TAB 3: INITIALIZATION ==========
            with gr.Tab("3️⃣ Initialize System", id=2):
                gr.Markdown("""
                ## 🔧 Step 3: Load Model Components
                Initialize ComfyUI nodes and prepare the generation pipeline.
                """)
                init_btn = gr.Button("⚡ Initialize Models", variant="primary", size="lg")
                init_logs = gr.Textbox(label="Initialization Logs", lines=15, interactive=False)
                init_status = gr.Markdown("⏳ Waiting to start...")
                next_to_generate = gr.Button("➡️ Next: Generate Videos", interactive=False, size="lg")
                init_btn.click(
                    fn=initialize_comfy_nodes,
                    outputs=[init_logs, next_to_generate]
                ).then(
                    fn=lambda: gr.update(value="✅ System initialized and ready!"),
                    outputs=[init_status]
                )
                next_to_generate.click(
                    fn=lambda: gr.update(selected=3),
                    outputs=[tabs]
                )
            # ========== TAB 4: VIDEO GENERATION ==========
            with gr.Tab("4️⃣ Generate Videos", id=3):
                gr.Markdown("""
                ## 🎨 Step 4: Create Your Video
                Configure parameters and generate your AI video!
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Prompts")
                        positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            value="a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera",
                            lines=4,
                            placeholder="Describe what you want to see..."
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="Bright tones, overexposure, static, blurry details, subtitles, artistic style, artwork, painting, still image, dull overall, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed limbs, finger fusion, static frame, messy background, three legs, crowded background, walking backwards",
                            lines=4,
                            placeholder="Describe what to avoid..."
                        )
                        gr.Markdown("### 🎬 Video Settings")
                        with gr.Row():
                            width = gr.Slider(256, 1280, 832, step=64, label="Width")
                            height = gr.Slider(256, 1280, 480, step=64, label="Height")
                        with gr.Row():
                            frames = gr.Slider(1, 120, 33, step=1, label="Frames (1 for single image)")
                            fps = gr.Slider(1, 60, 16, step=1, label="FPS")
                        gr.Markdown("### ⚙️ Generation Settings")
                        with gr.Row():
                            steps = gr.Slider(1, 100, 30, step=1, label="Steps (more = better quality, slower)")
                            cfg_scale = gr.Slider(1.0, 20.0, 1.0, step=0.1, label="CFG Scale")
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=82628696717253, precision=0)
                            use_random_seed = gr.Checkbox(label="Random Seed", value=False)
                        sampler_name = gr.Dropdown(
                            ["uni_pc", "euler", "dpmpp_2m", "ddim", "lms"],
                            value="uni_pc",
                            label="Sampler"
                        )
                        scheduler = gr.Dropdown(
                            ["simple", "normal", "karras", "exponential"],
                            value="simple",
                            label="Scheduler"
                        )
                        output_format = gr.Radio(
                            ["mp4", "webm"],
                            value="mp4",
                            label="Output Format"
                        )
                        generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 Output")
                        output_video = gr.Video(label="Generated Video", height=400)
                        generation_logs = gr.Textbox(label="Generation Logs", lines=10, interactive=False)
                gr.Markdown("### 💡 Example Prompts")
                gr.Examples(
                    examples=[
                        ["a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera", 832, 480, 33, 16],
                        ["ultra realistic car racing on a mountain road sunset golden hour, dynamic camera movement", 832, 480, 60, 24],
                        ["astronaut floating in deep space, earth in background, cinematic 4k", 832, 480, 45, 16],
                        ["peaceful lake with swans swimming, sunrise golden hour, calm water reflections", 832, 480, 50, 20],
                        ["cyberpunk city at night, neon lights, rain, flying cars, cinematic", 832, 480, 60, 24],
                    ],
                    inputs=[positive_prompt, width, height, frames, fps],
                )
                generate_btn.click(
                    fn=generate_video,
                    inputs=[
                        positive_prompt, negative_prompt, width, height, seed,
                        steps, cfg_scale, sampler_name, scheduler, frames, fps,
                        output_format, use_random_seed
                    ],
                    outputs=[output_video, generation_logs]
                )
    return app
# ==================== LAUNCH ====================
if __name__ == "__main__":
    print("🚀 Launching Wan 2.1 Video Generator...")
    app = create_interface()
    app.launch(
        share=True,
        debug=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7861
    )
    print("✅ Application launched successfully!")
    print("⚠️  Remember to restart runtime after Step 1 completes!")
