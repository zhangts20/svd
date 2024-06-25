import torch

from PIL import Image
from diffusers.utils import load_image, export_to_video


@torch.no_grad()
def pth_img2video_inference(
    model_dir: str,
    image_path: str,
    fps: int,
    num_frames: int,
    num_warmups: int = 1,
    output_path: str = "./out/pipeline.mp4",
) -> None:

    def load_model(model_dir: str):
        from diffusers import StableVideoDiffusionPipeline

        return StableVideoDiffusionPipeline.from_pretrained(model_dir).to(
            "cuda")

    def warmup(
        model: torch.nn.Module,
        image: Image.Image,
        num_warmups: int,
    ) -> None:
        for _ in range(num_warmups):
            model(image)

    def infer(
        model: torch.nn.Module,
        image: Image.Image,
        fps: int,
        num_frames: int,
    ) -> torch.Tensor:
        return model(image, fps=fps, num_frames=num_frames)

    def export(frames, out_path: str, fps: int):
        export_to_video(video_frames=frames,
                        output_video_path=out_path,
                        fps=fps)

    # load image
    image = load_image(image_path)
    image = image.resize((1024, 576))

    model = load_model(model_dir)
    warmup(model, image, num_warmups)

    frames = infer(model, image, fps, num_frames)
    export(frames, output_path, fps)


@torch.no_grad()
def img2video_inference(
    model_dir: str,
    image_path: str,
    fps: int,
    num_frames: int,
    num_warmups: int = 1,
    output_path: str = "./out/generate.mp4",
) -> None:

    def load_model(model_dir: str, config_name: str):
        from svd.core import Img2VideoPipeline

        return Img2VideoPipeline.from_pretrained(model_dir, config_name)

    def warmup(
        model: torch.nn.Module,
        image: Image.Image,
        num_warmups: int,
    ) -> None:
        for _ in range(num_warmups):
            model(image)

    def infer(
        model: torch.nn.Module,
        image: Image.Image,
        fps: int,
        num_frames: int,
    ) -> torch.Tensor:
        return model(image, fps=fps, num_frams=num_frames)

    def export(frames, out_path: str, fps: int = 7):
        export_to_video(video_frames=frames,
                        output_video_path=out_path,
                        fps=fps)

    # load image
    image = load_image(image_path)
    image = image.resize((1024, 576))

    model = load_model(model_dir, "model_index.json")
    warmup(model, image, num_warmups)

    frames = infer(model, image, fps, num_frames)
    export(frames, output_path)
