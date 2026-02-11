# Copyright Alibaba Inc. All Rights Reserved.
import warnings
warnings.filterwarnings('ignore')

import random
import argparse
from pathlib import Path
import json
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from utils import cameras_json_to_camera_list, get_intrinsic_matrix, save_video_imageio, \
    depth_to_world_coords_points, normalize_scene, \
    batch_depth_to_world, save_colored_pointcloud_ply, get_pointclouds
from FantasyWorld.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from FantasyWorld.diffsynth_wan22.data.dataset_re10k import RealEstate10KPoseProcessor
from FantasyWorld.fusion.model_wan22 import FantasyWorldFusionModel

sys.path.insert(0, "thirdparty/MoGe")
from moge.model.v2 import MoGeModel


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class FantasyWorldSampler:
    def __init__(
        self,
        ckpt_dir: str = "",
        model_ckpt_high: str = None,
        model_ckpt_low: str = None,
        base_seed: int = -1,
        sample_steps: int = 50,
        cfg_scale: float = 5.0,
        timestep_boundary: int = 900,
        frames: int = 81,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
    ):
        self.base_seed = base_seed if base_seed >= 0 else random.randint(
            0, sys.maxsize)
        self.sample_steps = sample_steps
        self.cfg_scale = cfg_scale
        self.fps = fps
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.num_frames = frames
        self.height = height
        self.width = width

        # Configuration for VGGT and camera control
        vggt_cfg = {
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": True,
            "enable_track": False,
            "DPT_patch_size": 16,
        }
        camera_cfg = {
            "pose_in_dim": 768,
            "plucker_fea_dim": 2048,
            "pose_inject_method": "adaln",
            "use_info": "plucker",
        }

        # Load high timestep model (1000-900)
        print("Loading HIGH timestep model (timestep 1000-900)...")
        print("  - Using high_noise_model DiT weights")
        print(
            f"  - LoRA: {ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors")
        self.model_high = FantasyWorldFusionModel(
            start_index=16,
            use_gradient_checkpointing=True,
            cross_attention_list=list(
                range(24)),
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            dit_path=ckpt_dir,
            lora_path=f"{ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors",
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
            load_vae=True,
            load_text_encoder=True)

        # Load low timestep model (900-0)
        print("Loading LOW timestep model (timestep 900-0)...")
        print("  - Using low_noise_model DiT weights")
        print(
            f"  - LoRA: {ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors")
        self.model_low = FantasyWorldFusionModel(
            start_index=16,
            use_gradient_checkpointing=True,
            cross_attention_list=list(
                range(24)),
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            dit_path=ckpt_dir,
            lora_path=f"{ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors",
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
        )

        # Move both models to GPU
        print("Moving both models to GPU...")
        self.model_high.to(torch.bfloat16)
        self.model_high.to(self.device)
        self.model_high.pipe.device = self.device
        self.model_high.eval()

        self.model_low.to(torch.bfloat16)
        self.model_low.to(self.device)
        self.model_low.pipe.device = self.device
        self.model_low.eval()
        print("Both models loaded to GPU successfully")

        # Load high model checkpoint
        print(f"Loading HIGH model checkpoint from: {model_ckpt_high}")
        ckpt_high = torch.load(model_ckpt_high, map_location=self.device)
        messages = self.model_high.load_state_dict(ckpt_high, strict=False)
        print("Missing keys = {}, Unexpected keys = {}".format(len(messages.missing_keys), len(messages.unexpected_keys)))
        assert not messages.unexpected_keys
        print("✓ Loaded HIGH model from checkpoint (direct state dict)")

        # Load low model checkpoint
        print(f"Loading LOW model checkpoint from: {model_ckpt_low}")
        ckpt_low = torch.load(model_ckpt_low, map_location=self.device)
        messages = self.model_low.load_state_dict(ckpt_low, strict=False)
        print("Missing keys = {}, Unexpected keys = {}".format(len(messages.missing_keys), len(messages.unexpected_keys)))
        assert not messages.unexpected_keys
        print("✓ Loaded LOW model from checkpoint (direct state dict)")

        self.pose_processor = RealEstate10KPoseProcessor(
            sample_stride=1,
            sample_n_frames=frames,
            relative_pose=True,
            zero_t_first_frame=True,
            sample_size=[height, width],
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            is_i2v=True,
        )
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device).eval()
        # Timestep boundary for switching between models
        self.timestep_boundary = timestep_boundary
        print(
            f"Timestep boundary set to: {self.timestep_boundary} (HIGH: 0-{self.timestep_boundary}, LOW: {self.timestep_boundary}-1000)")

    def generate_video_with_dual_models(
        self,
        context_pos: torch.Tensor,
        context_neg: torch.Tensor,
        y: torch.Tensor = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        sample_steps: int = 50,
        plucker_embedding: torch.Tensor = None,
    ):
        """
        Generate video using two separate models for high and low timesteps.
        High model: timestep 0-358
        Low model: timestep 358-1000
        """
        print("Starting dual-model video generation...")

        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1

        # Set timesteps for both models (they share the same scheduler)
        self.model_high.pipe.scheduler.set_timesteps(sample_steps)
        self.model_low.pipe.scheduler.set_timesteps(sample_steps)

        # Generate initial noise
        noise = self.model_high.pipe.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8), seed=self.base_seed)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        # Prepare camera control input if needed
        if self.model_high.camera_control:
            use_info = self.model_high.use_info
            if use_info == 'plucker':
                video_guidance = plucker_embedding
            else:
                raise NotImplementedError(
                    f"use_info={use_info} not implemented in inference")

            control_camera_video = video_guidance[0].permute(
                [3, 0, 1, 2]).unsqueeze(0)
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(
                b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(
                b, f // 4, c * 4, h, w).transpose(1, 2)
            control_camera_latents_input = control_camera_latents.to(
                device=self.device, dtype=self.torch_dtype)
        else:
            control_camera_latents_input = None

        image_emb = {}
        if y is not None:
            image_emb["y"] = y

        final_prediction = None

        # Denoising loop
        for progress_id, timestep in enumerate(tqdm(range(sample_steps))):
            t = self.model_high.pipe.scheduler.timesteps[progress_id].unsqueeze(
                0).to(dtype=self.torch_dtype, device=self.device)

            # Select model based on timestep value
            timestep_value = t.item()
            if timestep_value > self.timestep_boundary:
                # Use high model for timestep 1000-900
                current_model = self.model_high
            else:
                # Use low model for timestep 900-0
                current_model = self.model_low

            # Forward pass with selected model
            noise_pred_posi, prediction = current_model.joint_forward(
                latents,
                timestep=t,
                context=context_pos.to(dtype=self.torch_dtype, device=self.device),
                y=image_emb.get("y").to(dtype=self.torch_dtype, device=self.device),
                use_gradient_checkpointing=False,
                camera_token=None,
                control_camera_latents_input=control_camera_latents_input,
                uncond=False,
                return_prediction=True if progress_id == sample_steps - 1 else False,
            )
            # CFG if needed
            if self.cfg_scale != 1.0 and context_neg is not None:
                noise_pred_nega, _ = current_model.joint_forward(
                    latents,
                    timestep=t,
                    context=context_neg.to(dtype=self.torch_dtype, device=self.device),
                    y=image_emb.get("y").to(dtype=self.torch_dtype, device=self.device),
                    use_gradient_checkpointing=False,
                    camera_token=None,
                    control_camera_latents_input=control_camera_latents_input,
                    uncond=False,
                )
                noise_pred = noise_pred_nega + self.cfg_scale * \
                    (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Update latents
            latents = latents.to(self.device)
            latents = current_model.pipe.scheduler.step(
                noise_pred,
                current_model.pipe.scheduler.timesteps[progress_id],
                latents
            )

            # Store the last prediction
            final_prediction = prediction

        print("Dual-model video generation complete")
        return latents, final_prediction

    def generate_video(
        self,
        image_path,
        end_image_path,
        prompt,
        neg_prompt,
        camera_params,
        using_scale=True,
    ):
        print("Generating video ...")
        with torch.no_grad():
            input_image = Image.open(image_path).convert('RGB')
            input_image_pt = torch.tensor(
                np.array(input_image) / 255,
                dtype=torch.float32,
                device=self.device).permute(
                2,
                0,
                1)
            output = self.moge.infer(input_image_pt)
            moge = {k: v.cpu().contiguous() for k, v in output.items()}
            intrinsics = []
            extrinsics = []
            for camera in camera_params:
                intrinsics.append(get_intrinsic_matrix(camera))
                extrinsics.append(camera.w2c_mat)
            intrinsics = torch.from_numpy(
                np.stack(intrinsics).astype(np.float32))
            extrinsics = torch.from_numpy(
                np.stack(extrinsics).astype(np.float32))
            extrinsics_4x4 = extrinsics.unsqueeze(0)

            if using_scale:
                first_intrinsic = intrinsics[0, :, :].unsqueeze(0)
                first_extrinsic = extrinsics[0, :3, :].unsqueeze(0)
                first_moge_world, first_moge_mask = batch_depth_to_world(
                    prediction=moge, extrinsics=first_extrinsic, intrinsics=first_intrinsic)
                extrinsics_3x4 = extrinsics_4x4[:, :, :3, :]
                extrinsics = normalize_scene(
                    extrinsics=extrinsics_3x4,
                    first_moge_world=first_moge_world.unsqueeze(0),
                    first_moge_mask=first_moge_mask.unsqueeze(0),
                ).squeeze(0)

            image_hw = [self.height, self.width]

            pose_enc = extri_intri_to_pose_encoding(
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                image_hw,
                pose_encoding_type="absT_quaR_FoV").squeeze(0)
            plucker_embedding = self.pose_processor.get_plucker_embedding_direct_from_cam_params(
                pose_enc.unsqueeze(0), image_size=(
                    self.height, self.width)).to(
                self.device, self.torch_dtype)

            if end_image_path:
                end_image = Image.open(end_image_path)
            else:
                end_image = None
            inputs_shared, inputs_posi, inputs_nega = self.model_high.pipe(prompt=prompt, negative_prompt=neg_prompt,
                                                                           seed=self.base_seed, tiled=True,
                                                                           input_image=input_image,
                                                                           end_image=end_image,
                                                                           height=self.height, width=self.width,
                                                                           return_condition=True
                                                                           )
            ctx_pos, ctx_neg = inputs_posi["context"], inputs_nega["context"]
            y = inputs_shared["y"]

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.torch_dtype):
            latent_video, prediction = self.generate_video_with_dual_models(
                context_pos=ctx_pos,
                context_neg=ctx_neg,
                y=y,
                height=self.height,
                width=self.width,
                sample_steps=self.sample_steps,
                num_frames=self.num_frames,
                plucker_embedding=plucker_embedding,
            )
            frames = self.model_high.pipe.vae.decode(
                latent_video, device=self.device, tiled=True, tile_size=(
                    30, 52), tile_stride=(
                    15, 26))

        video = frames.squeeze(0)
        video = video.permute(1, 2, 3, 0)
        video = video.to(torch.float32).cpu()
        video = (video + 1.0) / 2.0
        video = (video * 255.0).clamp(0, 255)
        frames_np_processed = video.numpy().astype(np.uint8)

        return frames_np_processed, prediction


def main():
    parser = argparse.ArgumentParser(
        description="FantasyWorld Stage 2 Inference")
    # inputs
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        default="examples/images/input_image.png",
        help="Path to input image")
    parser.add_argument(
        "--end_image_path",
        type=str,
        default='',
        help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs.",
        help="Text prompt for generation")
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="Negative prompt for generation"
    )
    parser.add_argument(
        "--camera_json_path",
        type=str,
        required=True,
        default="example/cameras/camera_data.json",
        help="Path to camera data json")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=1.5,
        help="Confidence threshold for valid points in comparison")
    parser.add_argument(
        "--wan_ckpt_path",
        type=str,
        required=True,
        help="ckpt path")
    parser.add_argument(
        "--model_ckpt_high",
        type=str,
        required=True,
        help="Path to HIGH timestep model checkpoint file (.pth). Overrides model_ckpt for high model.")
    parser.add_argument(
        "--model_ckpt_low",
        type=str,
        required=True,
        help="Path to LOW timestep model checkpoint file (.pth). Overrides model_ckpt for low model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated video")
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Number of sampling steps")
    parser.add_argument(
        "--using_scale",
        type=str2bool,
        default=True,
        help="Whether to use scale normalization (True/False)")
    parser.add_argument(
        "--timestep_boundary",
        type=int,
        default=900,
        help="Timestep boundary between high and low models (default: 358)")
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480, matching training config)")
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832, matching training config)")
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed"
    )

    args = parser.parse_args()

    assert os.path.exists(args.image_path)
    if args.end_image_path:
        assert os.path.exists(args.end_image_path)
    camera_data = json.load(open(args.camera_json_path))
    camera_params = cameras_json_to_camera_list(camera_data, image_size=(args.height, args.width))

    print("Initializing FantasyWorld model...")
    model = FantasyWorldSampler(
        sample_steps=args.sample_steps,
        fps=args.fps,
        ckpt_dir=args.wan_ckpt_path,
        model_ckpt_high=args.model_ckpt_high,
        model_ckpt_low=args.model_ckpt_low,
        timestep_boundary=args.timestep_boundary,
        height=args.height,
        width=args.width,
        base_seed=args.seed,
    )
    video, prediction = model.generate_video(
        image_path=args.image_path,
        end_image_path=args.end_image_path,
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        camera_params=camera_params,
        using_scale=args.using_scale,
    )
    torch.save((video, prediction), "debug.pth")

    # Save output video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(args.image_path).stem
    output_path = output_dir / "video.mp4"

    save_video_imageio(video, output_path, fps=args.fps)
    print(f"Video saved to: {output_path}")

    recon_worldpoints = get_pointclouds(prediction, fix_first_frame=True)
    recon_ply_save_path = output_dir / \
        f"recon_confthresh{args.conf_threshold}.ply"

    valid_mask = prediction['depth_conf'] > args.conf_threshold
    # save pointcloud
    save_colored_pointcloud_ply(
        points=recon_worldpoints,
        colors=video,
        out_path=recon_ply_save_path,
        stride=4,
        max_points=None,
        valid_mask=valid_mask.cpu().numpy()[0]
    )
    print(f"3D prediction saved to: {recon_ply_save_path}")


if __name__ == "__main__":
    main()
