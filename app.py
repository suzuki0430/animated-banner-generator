import os
import boto3
import random
from PIL import Image
from rembg import remove
from moviepy import VideoFileClip
from moviepy.video.fx.Crop import Crop
import tempfile
import time
import uuid
import gradio as gr

# --- AWS settings from environment ---
s3_bucket = os.environ.get("S3_BUCKET", "generative-banners")
s3_input_banner_prefix = 'input/banners/'
s3_input_video_prefix = 'input/videos/'
s3_output_prefix = 'output/animated_banners/'
s3_region = os.environ.get("AWS_REGION", "ap-northeast-1")
s3 = boto3.client('s3', region_name=s3_region)


def remove_background(image_path, use_manual_transparency=False):
    if use_manual_transparency:
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        transparent_filename = f"{base_name}_transparent.png"
        transparent_path = os.path.join(
            os.path.dirname(image_path), transparent_filename)
        try:
            s3.download_file(
                s3_bucket, f"{s3_input_banner_prefix}transparent/{transparent_filename}", transparent_path)
            return transparent_path
        except Exception as e:
            print(f"Manual transparency fallback: {e}")

    input_image = Image.open(image_path)
    output_image = remove(input_image)
    transparent_path = image_path.replace('.', '_transparent.')
    output_image.save(transparent_path, 'PNG')
    return transparent_path


def extract_frames(video_path, max_frames=30, target_aspect_ratio=None):
    video = VideoFileClip(video_path)
    duration = video.duration
    fps = video.fps

    if target_aspect_ratio:
        original_width, original_height = video.size
        original_aspect = original_width / original_height
        target_w, target_h = map(int, target_aspect_ratio.split(':'))
        target_aspect = target_w / target_h

        if abs(original_aspect - target_aspect) > 0.01:
            if original_aspect > target_aspect:
                new_width = int(original_height * target_aspect)
                x1 = (original_width - new_width) // 2
                video = Crop(x1=x1, y1=0, width=new_width,
                             height=original_height).apply(video)
            else:
                new_height = int(original_width / target_aspect)
                y1 = (original_height - new_height) // 2
                video = Crop(x1=0, y1=y1, width=original_width,
                             height=new_height).apply(video)

    interval = duration / min(max_frames, int(duration * fps))
    frames = [video.get_frame(i * interval) for i in range(max_frames)]
    return frames, [i * interval for i in range(max_frames)], fps


def composite_banner_with_frame(banner_path, frame, output_path):
    banner = Image.open(banner_path).convert("RGBA")
    banner_width, banner_height = banner.size
    frame_pil = Image.fromarray(frame).convert("RGBA")
    frame_width, frame_height = frame_pil.size
    frame_aspect = frame_width / frame_height
    banner_aspect = banner_width / banner_height

    if frame_aspect > banner_aspect:
        new_height = banner_height
        new_width = int(new_height * frame_aspect)
        frame_pil = frame_pil.resize((new_width, new_height), Image.LANCZOS)
        left = (new_width - banner_width) // 2
        frame_pil = frame_pil.crop(
            (left, 0, left + banner_width, banner_height))
    else:
        new_width = banner_width
        new_height = int(new_width / frame_aspect)
        frame_pil = frame_pil.resize((new_width, new_height), Image.LANCZOS)
        top = (new_height - banner_height) // 2
        frame_pil = frame_pil.crop((0, top, banner_width, top + banner_height))

    result = frame_pil.copy()
    result.paste(banner, (0, 0), banner)
    result.convert("RGB").save(output_path, "JPEG")
    return output_path


def create_gif_from_frames(frame_paths, output_gif_path, fps):
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(output_gif_path, format='GIF', append_images=frames[1:],
                   save_all=True, duration=int(1000/fps), loop=0)
    return output_gif_path


def generate_nova_reel_video(prompt, duration=6):
    temp_dir = tempfile.mkdtemp()
    output_video_path = os.path.join(temp_dir, f"nova_reel_{uuid.uuid4()}.mp4")
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=s3_region)
        output_uri = f"s3://{s3_bucket}/{s3_output_prefix}nova_output/"
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {"text": prompt},
            "videoGenerationConfig": {
                "fps": 24,
                "durationSeconds": min(duration, 6),
                "dimension": "1280x720",
                "seed": random.randint(0, 2147483646)
            }
        }
        response = bedrock.start_async_invoke(
            modelId="amazon.nova-reel-v1:0",
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}}
        )
        arn = response["invocationArn"]
        for _ in range(20):
            status = bedrock.get_async_invoke(invocationArn=arn)["status"]
            if status == "Completed":
                response = s3.list_objects_v2(
                    Bucket=s3_bucket, Prefix=f"{s3_output_prefix}nova_output/")
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('output.mp4'):
                        s3.download_file(
                            s3_bucket, obj['Key'], output_video_path)
                        return output_video_path
            elif status == "Failed":
                raise Exception("Nova Reel generation failed")
            time.sleep(30)
        raise Exception("Nova Reel generation timed out")
    except Exception:
        fallback_path = os.path.join(temp_dir, "demo_video.mp4")
        s3.download_file(
            s3_bucket, f"{s3_input_video_prefix}demo_video.mp4", fallback_path)
        return fallback_path


def create_animated_banner(banner_file, video_prompt=None, video_file=None, output_name=None, max_frames=20, use_manual_transparency=False):
    start = time.time()
    output_name = output_name or f"animated_banner_{int(start)}"
    temp_dir = tempfile.mkdtemp()
    local_banner_path = os.path.join(temp_dir, os.path.basename(banner_file))
    s3.download_file(
        s3_bucket, f"{s3_input_banner_prefix}{banner_file}", local_banner_path)
    local_video_path = generate_nova_reel_video(
        video_prompt) if video_prompt else os.path.join(temp_dir, os.path.basename(video_file))
    if video_file and not video_prompt:
        s3.download_file(
            s3_bucket, f"{s3_input_video_prefix}{video_file}", local_video_path)
    banner = Image.open(local_banner_path)
    aspect = f"{banner.width}:{banner.height}"
    transparent_path = remove_background(
        local_banner_path, use_manual_transparency)
    frames, _, fps = extract_frames(
        local_video_path, max_frames, target_aspect_ratio=aspect)
    composite_paths = [composite_banner_with_frame(transparent_path, f, os.path.join(
        temp_dir, f"frame_{i:03}.jpg")) for i, f in enumerate(frames)]
    gif_path = os.path.join(temp_dir, f"{output_name}.gif")
    gif_path = create_gif_from_frames(composite_paths, gif_path, fps/2)
    s3_key = f"{s3_output_prefix}{output_name}.gif"
    s3.upload_file(gif_path, s3_bucket, s3_key)
    return gif_path, s3_key


def gradio_workflow(banner_image, video_prompt, use_manual_transparency, status_box):
    try:
        status_box.append("🔁 Uploading banner image...")
        tmp_path = "/tmp/gradio_banner.png"
        banner_image.save(tmp_path)
        s3_key = f"{s3_input_banner_prefix}gradio_uploaded_banner.png"
        s3.upload_file(tmp_path, s3_bucket, s3_key)
        status_box.append("✨ Creating animated banner...")
        gif_path, gif_s3_key = create_animated_banner(
            banner_file="gradio_uploaded_banner.png",
            video_prompt=video_prompt,
            output_name="gradio_generated_banner",
            use_manual_transparency=use_manual_transparency
        )
        gif_url = f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/{gif_s3_key}"
        status_box.append("✅ Done!")
        return gif_path, gif_url
    except Exception as e:
        status_box.append(f"❌ Error: {str(e)}")
        return None, None


with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Create Animated Banner with Amazon Nova Reel")
    with gr.Row():
        with gr.Column():
            banner_input = gr.Image(type="pil", label="📌 Upload Banner Image")
            prompt_input = gr.Textbox(
                label="🎨 Video Prompt", placeholder="e.g. Radiant golden light...")
            transparency_checkbox = gr.Checkbox(
                label="Use Manual Transparency", value=True)
            run_button = gr.Button("🚀 Generate")
        with gr.Column():
            status_output = gr.Textbox(label="Progress Log", lines=8)
            gif_output = gr.Image(
                type="filepath", label="🎞️ Generated GIF Preview")
            link_output = gr.Textbox(label="🔗 S3 Output URL")
    status_box = gr.State([])

    def append_status(msg, history):
        history.append(msg)
        return "\n".join(history), history

    def run_and_display(banner_image, video_prompt, use_manual_transparency, history):
        _, history = append_status("🟢 Starting process...", history)
        gif_path, gif_url = gradio_workflow(
            banner_image, video_prompt, use_manual_transparency, history)
        status_text = "\n".join(history)
        return gif_path, gif_url, status_text, history

    run_button.click(
        fn=run_and_display,
        inputs=[banner_input, prompt_input, transparency_checkbox, status_box],
        outputs=[gif_output, link_output, status_output, status_box]
    )

if __name__ == "__main__":
    demo.launch()
