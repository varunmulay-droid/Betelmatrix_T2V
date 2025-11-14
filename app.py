# Complete Gradio Application for Wan 2.1 Video Generation
# Run this in Google Colab
# GitHub Ready: https://github.com/yourusername/wan2.1-video-gen

import gradio as gr
import subprocess
import sys
import os
import torch
import gc
import random
import numpy as np
from PIL import Image
import imageio
from pathlib import Path

# ==================== GLOBAL STATE ====================
class AppState:
    def __init__(self):
        self.env_setup_complete = False
        self.models_downloaded = False
        self.models_loaded = False
        self.comfy_path = "/content/ComfyUI"
        self.output_dir = "/content/Betelmatrix_AI_video_Gen/output"
        self.useQ6 = False
       
        # Model components
        self.clip = None
        self.model = None
        self.vae = None
       
        # Node instances
        self.unet_loader = None
        self.clip_loader = None
        self.clip_encode_positive = None
        self.clip_encode_negative = None
        self.vae_loader = None
        self.empty_latent_video = None
        self.ksampler = None
        self.vae_decode = None

state = AppState()
os.makedirs(state.output_dir, exist_ok=True)

# ==================== ENVIRONMENT SETUP ====================
def install_dependencies(progress=gr.Progress()):
    """Step 1: Install all required dependencies"""
    logs = []
    try:
        progress(0.1, desc="Installing PyTorch...")
        logs.append("Installing PyTorch 2.6.0...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                       "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu121"],
                       check=True, capture_output=True)
        logs.append("PyTorch installed")

        progress(0.3, desc="Installing core packages...")
        logs.append("Installing torchsde, einops, diffusers...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                       "torchsde", "einops", "diffusers", "accelerate",
                       "xformers==0.0.29.post2", "av", "safetensors", "transformers"],
                       check=True, capture_output=True)
        logs.append("Core packages installed")

        progress(0.5, desc="Cloning ComfyUI...")
        if not os.path.exists("/content/ComfyUI"):
            logs.append("Cloning ComfyUI repository...")
            subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", "/content/ComfyUI"],
                          check=True, capture_output=True)
            logs.append("ComfyUI cloned")
        else:
            logs.append("ComfyUI already exists")

        progress(0.7, desc="Setting up custom nodes...")
        custom_nodes_path = "/content/ComfyUI/custom_nodes"
        os.makedirs(custom_nodes_path, exist_ok=True)

        gguf_path = f"{custom_nodes_path}/ComfyUI_GGUF"
        if not os.path.exists(gguf_path):
            logs.append("Cloning ComfyUI_GGUF...")
            subprocess.run(["git", "clone", "https://github.com/city96/ComfyUI-GGUF", gguf_path],
                          check=True, capture_output=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{gguf_path}/requirements.txt"],
                          check=True, capture_output=True)
            logs.append("ComfyUI_GGUF installed")
        else:
            logs.append("ComfyUI_GGUF already exists")

        progress(0.9, desc="Installing system packages...")
        logs.append("Installing aria2 and ffmpeg...")
        subprocess.run(["apt-get", "update", "-qq"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "aria2", "ffmpeg"], check=True, capture_output=True)
        logs.append("System packages installed")

        progress(1.0, desc="Complete!")
        state.env_setup_complete = True
        logs.append("Environment setup complete!")

        return "\n".join(logs), gr.update(interactive=True)

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return "\n".join(logs), gr.update(interactive=False)


def download_models(use_q6, progress=gr.Progress()):
    """Step 2: Download model files"""
    logs = []
    state.useQ6 = use_q6

    try:
        os.makedirs("/content/ComfyUI/models/unet", exist_ok=True)
        os.makedirs("/content/ComfyUI/models/text_encoders", exist_ok=True)
        os.makedirs("/content/ComfyUI/models/vae", exist_ok=True)

        progress(0.1, desc="Downloading UNET model...")
        model_name = "wan2.1-t2v-14b-Q6_K.gguf" if use_q6 else "wan2.1-t2v-14b-Q5_0.gguf"
        model_url = f"https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/{model_name}"
        model_path = f"/content/ComfyUI/models/unet/{model_name}"

        if not os.path.exists(model_path):
            logs.append(f"Downloading {model_name}...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M",
                           model_url, "-d", "/content/ComfyUI/models/unet", "-o", model_name], check=True)
            logs.append(f"{model_name} downloaded")
        else:
            logs.append(f"{model_name} already exists")

        progress(0.4, desc="Downloading text encoder...")
        text_encoder_path = "/content/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        if not os.path.exists(text_encoder_path):
            logs.append("Downloading text encoder...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M",
                           "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                           "-d", "/content/ComfyUI/models/text_encoders", "-o", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
                           check=True)
            logs.append("Text encoder downloaded")
        else:
            logs.append("Text encoder already exists")

        progress(0.7, desc="Downloading VAE...")
        vae_path = "/content/ComfyUI/models/vae/wan_2.1_vae.safetensors"
        if not os.path.exists(vae_path):
            logs.append("Downloading VAE...")
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M",
                           "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
                           "-d", "/content/ComfyUI/models/vae", "-o", "wan_2.1_vae.safetensors"], check=True)
            logs.append("VAE downloaded")
        else:
            logs.append("VAE already exists")

        progress(1.0, desc="Complete!")
        state.models_downloaded = True
        logs.append("All models downloaded successfully!")

        return "\n".join(logs), gr.update(interactive=True)

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return "\n".join(logs), gr.update(interactive=False)


# ==================== UTILITY FUNCTIONS ====================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_as_mp4(images, filename_prefix, fps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"
    frames = [(img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8) for img in images]
    with imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path


def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=10, output_dir=None):
    if output_dir is None:
        output_dir = state.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webm"
    frames = [(img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8) for img in images]
    with imageio.get_writer(output_path, fps=fps, codec=codec, quality=quality, format='webm') as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path


def save_as_image(image, filename_prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"
    frame = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(frame).save(output_path)
    return output_path


# ==================== MODEL LOADING ====================
def initialize_comfy_nodes(progress=gr.Progress()):
    """Step 3: Initialize ComfyUI nodes"""
    logs = []
    try:
        progress(0.1, desc="Adding ComfyUI to path...")
        sys.path.insert(0, '/content/ComfyUI')
        logs.append("ComfyUI path configured")

        progress(0.3, desc="Importing ComfyUI modules...")
        try:
            from nodes import NODE_CLASS_MAPPINGS
            CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]
            CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]
            VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]
            VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]
            KSampler = NODE_CLASS_MAPPINGS["KSampler"]
        except Exception as e:
            logs.append(f"Failed to import standard nodes: {e}")
            return "\n".join(logs), gr.update(interactive=False)

        try:
            from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
        except Exception as e:
            logs.append(f"Failed to import GGUF node: {e}")
            return "\n".join(logs), gr.update(interactive=False)

        try:
            from comfy_extras.nodes_latent import EmptyLatentImage
            EmptyHunyuanLatentVideo = EmptyLatentImage  # Fallback
        except:
            try:
                from nodes import NODE_CLASS_MAPPINGS
                EmptyHunyuanLatentVideo = NODE_CLASS_MAPPINGS.get("EmptyLatentImage", None)
            except:
                logs.append("EmptyLatentImage not found")
                return "\n".join(logs), gr.update(interactive=False)

        logs.append("ComfyUI modules imported")

        progress(0.6, desc="Initializing nodes...")
        state.unet_loader = UnetLoaderGGUF()
        state.clip_loader = CLIPLoader()
        state.clip_encode_positive = CLIPTextEncode()
        state.clip_encode_negative = CLIPTextEncode()
        state.vae_loader = VAELoader()
        state.empty_latent_video = EmptyHunyuanLatentVideo()
        state.ksampler = KSampler()
        state.vae_decode = VAEDecode()

        logs.append("All nodes initialized")
        progress(1.0, desc="Complete!")
        state.models_loaded = True
        logs.append("Model loading system ready!")

        return "\n".join(logs), gr.update(interactive=True)

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return "\n".join(logs), gr.update(interactive=False)


# ==================== VIDEO GENERATION ====================
def generate_video(
    positive_prompt, negative_prompt, width, height, seed, steps,
    cfg_scale, sampler_name, scheduler, frames, fps, output_format,
    use_random_seed, progress=gr.Progress()
):
    if not state.models_loaded:
        return None, "Please complete setup steps first!"

    logs = []
    try:
        if use_random_seed:
            seed = random.randint(0, 2**32 - 1)
            logs.append(f"Random seed: {seed}")

        with torch.inference_mode():
            progress(0.1, desc="Loading text encoder...")
            clip = state.clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]

            progress(0.2, desc="Encoding prompts...")
            positive = state.clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = state.clip_encode_negative.encode(clip, negative_prompt)[0]

            del clip
            clear_memory()

            progress(0.3, desc="Creating latent space...")
            empty_latent = state.empty_latent_video.generate(width=width, height=height, batch_size=1)[0]

            progress(0.4, desc="Loading UNET model...")
            model_name = "wan2.1-t2v-14b-Q6_K.gguf" if state.useQ6 else "wan2.1-t2v-14b-Q5_0.gguf"
            model = state.unet_loader.load_unet(model_name)[0]

            progress(0.5, desc="Generating video...")
            sampled = state.ksampler.sample(
                model=model, seed=seed, steps=steps, cfg=cfg_scale,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative=negative, latent_image=empty_latent
            )[0]

            del model
            clear_memory()

            progress(0.8, desc="Loading VAE...")
            vae = state.vae_loader.load_vae("wan_2.1_vae.safetensors")[0]

            progress(0.9, desc="Decoding latents...")
            decoded = state.vae_decode.decode(vae, sampled)[0]["samples"]

            del vae
            clear_memory()

            progress(0.95, desc="Saving output...")
            timestamp = random.randint(1000, 9999)
            output_path = ""

            if frames == 1:
                output_path = save_as_image(decoded[0], f"frame_{timestamp}", state.output_dir)
            else:
                if output_format.lower() == "webm":
                    output_path = save_as_webm(decoded, f"video_{timestamp}", fps, quality=10)
                else:
                    output_path = save_as_mp4(decoded, f"video_{timestamp}", fps, state.output_dir)

            progress(1.0, desc="Complete!")
            logs.append(f"Generation complete! Seed: {seed}")
            logs.append(f"Saved to: {output_path}")

            return output_path, "\n".join(logs)

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        clear_memory()
        return None, "\n".join(logs)


# ==================== GRADIO INTERFACE ====================
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        title="Betelmatrix - Wan2.1 AI Video Generator",
        css="""
        .gradio-container { max-width: 1400px !important; }
        .step-complete { background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%); padding: 20px; border-radius: 10px; margin: 10px 0; }
        """
    ) as app:

        gr.Markdown("# Betelmatrix : Wan2.1 AI Video Generator\n### Create stunning AI videos with 14B diffusion model")

        with gr.Tabs() as tabs:
            # TAB 1
            with gr.Tab("1. Environment Setup", id=0):
                gr.Markdown("## Step 1: Install Dependencies\nInstall PyTorch, ComfyUI, and packages (~5-10 min).")
                setup_btn = gr.Button("Start Installation", variant="primary")
                setup_logs = gr.Textbox(label="Logs", lines=15, interactive=False)
                setup_status = gr.Markdown("Waiting...")
                next_to_download = gr.Button("Next: Download Models", interactive=False)

                setup_btn.click(install_dependencies, outputs=[setup_logs, next_to_download]) \
                         .then(lambda: gr.update(value="Environment ready!"), outputs=setup_status)
                next_to_download.click(lambda: gr.update(selected=1), outputs=tabs)

            # TAB 2
            with gr.Tab("2. Download Models", id=1):
                gr.Markdown("## Step 2: Download Models (~15-20 GB)")
                use_q6_checkbox = gr.Checkbox(label="Use Q6_K (Higher VRAM)", value=False)
                download_btn = gr.Button("Download Models", variant="primary")
                download_logs = gr.Textbox(label="Logs", lines=15, interactive=False)
                download_status = gr.Markdown("Waiting...")
                next_to_init = gr.Button("Next: Initialize", interactive=False)

                download_btn.click(download_models, inputs=use_q6_checkbox, outputs=[download_logs, next_to_init]) \
                            .then(lambda: gr.update(value="Models ready!"), outputs=download_status)
                next_to_init.click(lambda: gr.update(selected=2), outputs=tabs)

            # TAB 3
            with gr.Tab("3. Initialize System", id=2):
                gr.Markdown("## Step 3: Load Pipeline")
                init_btn = gr.Button("Initialize Models", variant="primary")
                init_logs = gr.Textbox(label="Logs", lines=15, interactive=False)
                init_status = gr.Markdown("Waiting...")
                next_to_generate = gr.Button("Next: Generate", interactive=False)

                init_btn.click(initialize_comfy_nodes, outputs=[init_logs, next_to_generate]) \
                        .then(lambda: gr.update(value="System ready!"), outputs=init_status)
                next_to_generate.click(lambda: gr.update(selected=3), outputs=tabs)

            # TAB 4
            with gr.Tab("4. Generate Videos", id=3):
                gr.Markdown("## Step 4: Create Your Video")
                with gr.Row():
                    with gr.Column():
                        positive_prompt = gr.Textbox(label="Positive Prompt", lines=4,
                            value="a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera")
                        negative_prompt = gr.Textbox(label="Negative Prompt", lines=4,
                            value="blurry, static, low quality, artifacts, text, watermark")

                        with gr.Row():
                            width = gr.Slider(256, 1280, 832, step=64, label="Width")
                            height = gr.Slider(256, 1280, 480, step=64, label="Height")
                        with gr.Row():
                            frames = gr.Slider(1, 120, 33, step=1, label="Frames")
                            fps = gr.Slider(1, 60, 16, step=1, label="FPS")
                        with gr.Row():
                            steps = gr.Slider(1, 100, 30, step=1, label="Steps")
                            cfg_scale = gr.Slider(1.0, 20.0, 1.0, step=0.1, label="CFG Scale")
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=12345, precision=0)
                            use_random_seed = gr.Checkbox(label="Random Seed", value=True)
                        sampler_name = gr.Dropdown(["euler", "dpmpp_2m", "uni_pc"], value="uni_pc", label="Sampler")
                        scheduler = gr.Dropdown(["simple", "karras"], value="simple", label="Scheduler")
                        output_format = gr.Radio(["mp4", "webm"], value="mp4", label="Format")
                        generate_btn = gr.Button("Generate Video", variant="primary")

                    with gr.Column():
                        output_video = gr.Video(label="Output", height=400)
                        generation_logs = gr.Textbox(label="Logs", lines=10, interactive=False)

                generate_btn.click(
                    generate_video,
                    inputs=[positive_prompt, negative_prompt, width, height, seed, steps, cfg_scale,
                            sampler_name, scheduler, frames, fps, output_format, use_random_seed],
                    outputs=[output_video, generation_logs]
                )

    return app


# ==================== LAUNCH ====================
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.0"], check=True)
    app = create_interface()
    app.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7861,
        allowed_paths=["/content/Betelmatrix_AI_video_Gen/output", "/content/ComfyUI/output"]
    )
