import os
import argparse

from svd.core import img2video_inference


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svd-dir",
        required=True,
        type=str,
        help="The root directory of stable video diffusion model.")
    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Whether to use TensorRT for the inference of UNet.")
    parser.add_argument("--in-img-path",
                        type=str,
                        required=True,
                        help="The path of input image.")
    parser.add_argument("--out-video-path",
                        type=str,
                        default="out/generated.mp4",
                        help="The path of generated video.")
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Whether to use pipeline to inference, out is `out/pipeline.mp4`."
    )
    parser.add_argument("--fps",
                        type=int,
                        default=7,
                        help="The Frames Per Second of generated video.")
    parser.add_argument("--num-frames",
                        type=int,
                        default=25,
                        help="The number of frames of generated video.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    dirname = os.path.dirname(os.path.join(os.getcwd(), args.out_img_path))
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if args.use_pipeline:
        from svd.core import pth_img2video_inference
        pth_img2video_inference(args.svd_dir,
                                args.in_img_path,
                                fps=args.fps,
                                num_frames=args.num_frames,
                                output_path="./out/pipeline.mp4")

    img2video_inference(args.svd_dir,
                        args.in_img_path,
                        fps=args.fps,
                        num_frames=args.num_frames,
                        output_path=args.out_video_path)
