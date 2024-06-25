import os
import torch
import json
import inspect
import importlib

from PIL import Image
from typing import Any, Dict
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import EulerDiscreteScheduler, UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from diffusers.video_processor import VideoProcessor


class Img2VideoPipeline:

    def __init__(
        self,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        scheduler: EulerDiscreteScheduler,
        unet: UNetSpatioTemporalConditionModel,
        vae: AutoencoderKLTemporalDecoder,
        engien_path: str = None,
        use_trt: bool = False,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.image_encoder = image_encoder
        self.scheduler = scheduler

        if use_trt:
            pass
        else:
            self.unet = unet

        self.vae = vae
        self.device = "cuda"
        self.dtype = torch.float16
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) -
                                    1)
        self.video_processor = VideoProcessor(
            do_resize=True, vae_scale_factor=self.vae_scale_factor)
        self.use_trt = use_trt

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        config_name: str,
                        use_trt: bool = False) -> "Img2VideoPipeline":
        # Parse config file.
        config_path = os.path.join(model_dir, config_name)
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            config_dict: Dict = json.load(f)
        # Get expected modules.
        expected_moduels = inspect.signature(cls.__init__).parameters
        config_dict = {
            k: v
            for k, v in config_dict.items() if k in expected_moduels
        }
        # e.g., name=feature_extractor, library_name=transformes, class_name=CLIPImageProcessor
        init_kwargs: Dict[str, Any] = dict()
        for name, (library_name, class_name) in config_dict.items():
            module = importlib.import_module(library_name)
            cls_ = getattr(module, class_name)
            # Class from_pretrained to initialize module.
            init_kwargs[name] = cls_.from_pretrained(
                os.path.join(model_dir, name))

        if use_trt:
            pass

        return cls(**init_kwargs)

    def cuda(self):
        self.text_encoder = self.text_encoder.to(self.device)
        self.image_encoder = self.image_encoder.to(self.device)

        if not self.use_trt:
            self.unet = self.unet.to(self.device)

        self.vae = self.vae.to(self.device)

        return self

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        fps: int,
        num_frames: int,
        noise_aug_strength: float = 0.02,
        motion_bucket_id: int = 127,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):
        # TODO:
        do_classifier_guidance = False

        # Encode input images. TODO
        image_embeddings = self.encode_image(image,
                                             do_classifier_free_guidance=False)

        # TODO:
        fps = fps - 1

        # Encode input image using VAE.
        image: torch.Tensor = self.video_processor.preprocess(
            image,
            height=self.unet.config.sample_size * self.vae_scale_factor,
            width=self.unet.config.sample_size * self.vae_scale_factor)

        # Add noise.
        noise = torch.randn(image.shape).to(device=self.device,
                                            dtype=image.dtype)
        image = image + noise_aug_strength * noise

        image_latents = self.encode_vae_image(
            image, do_classifier_free_guidance=False)

        # Repeat the image latents for each frame so we can cat them with the
        # noise. (bs, c, h, w) -> (bs, num_frames, c, h, w)
        image_latents = image_latents.unsqueeze(1).repeat(
            1, num_frames, 1, 1, 1)

        # Get added time ids.
        added_time_ids = self.get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            do_classifier_free_guidance=False)
        added_time_ids = added_time_ids.to(self.device)

        # Prepare timesteps.
        timesteps = self.retrieve_timesteps(self.scheduler,
                                            num_inference_steps)

        # Prepare latent variables.
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            1, num_frames, num_channels_latents,
            self.unet.config.sample_size * self.vae_scale_factor,
            self.unet.config.sample_size * self.vae_scale_factor)

        # Prepare guidance scale.
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale,
                                        num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device=self.device,
                                           dtype=latents.dtype)
        dims_to_append = latents.ndim - guidance_scale.ndim
        guidance_scale = guidance_scale[(..., ) + (None, ) * dims_to_append]

        # Denosing.
        for _, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance.
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # Concatenate image latents over channels dimension.
            latent_model_input = torch.cat([latent_model_input, image_latents],
                                           dim=2)

            # Predict the noise residual.
            noise_pred: torch.Tensor = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            # Perform guidance.
            if do_classifier_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale(
                    noise_pred_cond - noise_pred_uncond)

            # Compute the previous noisy sample.
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def encode_image(
        self,
        image: Image.Image,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        # Normalize the image with for CLIP input.
        image: torch.Tensor = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt").pixel_values
        image = image.to(device=self.device, dtype=self.dtype)
        image_embeddings: torch.Tensor = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # Add negative embeddings.
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat(
                [negative_image_embeddings, image_embeddings])

        return image_embeddings

    def encode_vae_image(
        self,
        image: torch.Tensor,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        image_latents: torch.Tensor = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
            add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if passed_add_embed_dim != expected_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def retrieve_timesteps(self, scheduler: EulerDiscreteScheduler,
                           num_inference_steps: int) -> torch.Tensor:
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        return timesteps

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channel_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        shape = (batch_size, num_frames, num_channel_latents // 2,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = torch.randn(shape, device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        return latents
