# Copyright Alibaba Inc. All Rights Reserved.

from ..vggt.models.vggt import VGGT
from ..diffsynth_wan22.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from ..diffsynth_wan22.models.wan_video_dit import sinusoidal_embedding_1d
import torch
import random
import torch.nn as nn
from ..fusion.layer.block import IRGBlock
from typing import Optional
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import copy
from ..diffsynth_wan22.models.wan_video_dit import build_freqs_3d_with_extra_cis, precompute_freqs_cis_3d
import torch
from einops import rearrange


def load_lora(pipeline, lora_path, multiplier, sub_transformer_name):
    from collections import defaultdict
    from safetensors.torch import load_file
    state_dict = load_file(lora_path)
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    device = pipeline.device
    dtype = pipeline.torch_dtype
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            key = "lora_unet__" + key
        key = key.replace(".", "_")
        if key.endswith("_lora_up_weight"):
            key = key[:-15] + ".lora_up.weight"
        if key.endswith("_lora_down_weight"):
            key = key[:-17] + ".lora_down.weight"
        if key.endswith("_lora_A_default_weight"):
            key = key[:-21] + ".lora_A.weight"
        if key.endswith("_lora_B_default_weight"):
            key = key[:-21] + ".lora_B.weight"
        if key.endswith("_lora_A_weight"):
            key = key[:-14] + ".lora_A.weight"
        if key.endswith("_lora_B_weight"):
            key = key[:-14] + ".lora_B.weight"
        if key.endswith("_alpha"):
            key = key[:-6] + ".alpha"
        key = key.replace(".lora_A.default.", ".lora_down.")
        key = key.replace(".lora_B.default.", ".lora_up.")
        key = key.replace(".lora_A.", ".lora_down.")
        key = key.replace(".lora_B.", ".lora_up.")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    for layer, elems in updates.items():
        layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
        curr_layer = getattr(pipeline, sub_transformer_name)
        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(
                            temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(
                                    f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                layer_infos = layer.split(
                    LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                curr_layer = getattr(pipeline, sub_transformer_name)
                len_layer_infos = len(layer_infos)
                start_index = 0 if len_layer_infos >= 1 and len(
                    layer_infos[0]) > 0 else 1
                end_indx = len_layer_infos
                error_flag = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(
                                f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__(
                            "_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue
        origin_dtype = curr_layer.weight.data.dtype
        origin_device = curr_layer.weight.data.device
        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * \
                alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)


class FantasyWorldFusionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 start_index=16,
                 use_gradient_checkpointing=True,
                 use_gradient_checkpointing_offload=False,
                 cross_attention_list=[0],
                 dit_path=None,
                 lora_path=None,
                 origin_file_pattern=None,
                 model_id="PAI/Wan2.2-Fun-A14B-Control-Camera",
                 vggt_cfg: dict | None = None,
                 camera_control: bool = False,
                 camera_cfg: dict | None = None,
                 min_timestep_boundary=0,
                 max_timestep_boundary=1,
                 load_vae=False,
                 load_text_encoder=False,
                 ):
        super().__init__()
        # freq_dim : the dimension for the sinusoidal_embedding_1d, default using 256
        # proj_mode: choosing from conv and linear module
        self.device = "cuda"
        print("Initializing WanVideoPipeline...")
        if load_vae and load_text_encoder:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=[
                    ModelConfig(
                        model_id=model_id,
                        origin_file_pattern=origin_file_pattern,
                        local_model_path=dit_path),
                    # ModelConfig(model_id=model_id, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", local_model_path=dit_path),
                    ModelConfig(
                        model_id=model_id,
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        local_model_path=dit_path),
                    ModelConfig(
                        model_id=model_id,
                        origin_file_pattern="Wan2.1_VAE.pth",
                        local_model_path=dit_path),
                ],
            )
        else:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=[
                    ModelConfig(
                        model_id=model_id,
                        origin_file_pattern=origin_file_pattern,
                        local_model_path=dit_path),
                    # ModelConfig(model_id=model_id, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", local_model_path=dit_path),
                    # ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", local_model_path=dit_path),
                    # ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", local_model_path=dit_path),
                ],
                tokenizer_config=None,
            )
        print("Initialized WanVideoPipeline.")
        pipe.device = "cpu"

        load_lora(pipe, lora_path, 0.55, "dit")

        self.pipe = pipe
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.min_timestep_boundary = min_timestep_boundary
        self.max_timestep_boundary = max_timestep_boundary
        print("initializing VGGT...")
        self.vggt = VGGT(**vggt_cfg)
        self.vggt.to(torch.bfloat16)
        print("initialized VGGT.")
        self.camera_control = camera_control

        self.start_index = start_index
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.cross_attention_list = cross_attention_list

        print("initializing IRGBlock...")
        irg_blocks = nn.ModuleList()
        self.bicross_dim = 1152
        self.bicross_num_heads = 12
        head_dim = self.bicross_dim // self.bicross_num_heads

        self.freqs_bicross = precompute_freqs_cis_3d(head_dim)

        for idx in self.cross_attention_list:
            src_dit_blk = self.pipe.dit.blocks[idx + self.start_index]
            src_agg_blk = self.vggt.aggregator.global_blocks[idx]
            dit_blk_copy = copy.deepcopy(src_dit_blk)
            agg_blk_copy = copy.deepcopy(src_agg_blk)
            self.pipe.dit.blocks[idx + self.start_index] = nn.Identity()
            self.vggt.aggregator.global_blocks[idx] = nn.Identity()
            irg_blocks.append(
                IRGBlock(
                    x_dit_block=dit_blk_copy,
                    x_agg_block=agg_blk_copy,
                    m1_dim=self.pipe.dit.dim,
                    m2_dim=self.vggt.embed_dim,
                    hidden_size=self.bicross_dim,
                    num_heads=self.bicross_num_heads,
                    drop_path=None,
                )
            )
        self.IRGBlock = irg_blocks
        print("initialized IRGBlock.")

        self.use_info = camera_cfg['use_info']
        self = self.to(dtype=torch.bfloat16)

    def joint_forward(self,
                      x: torch.Tensor,
                      timestep: torch.Tensor,
                      context: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      use_gradient_checkpointing: bool = True,
                      camera_token=None,
                      control_camera_latents_input: Optional[torch.Tensor] = None,
                      uncond=False,
                      return_prediction=False,
                      **kwargs,
                      ):
        f = x.shape[2]

        t = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep))
        t_mod = self.pipe.dit.time_projection(
            t).unflatten(1, (6, self.pipe.dit.dim))
        context = self.pipe.dit.text_embedding(context)

        if y is not None and self.pipe.dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)

        x = self.pipe.dit.patchify(x, control_camera_latents_input)
        f, h, w = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

        freqs = torch.cat([
            self.pipe.dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.pipe.dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.pipe.dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs_bi_dit = torch.cat([
            self.freqs_bicross[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs_bicross[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs_bicross[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs_bi_agg = build_freqs_3d_with_extra_cis(
            self.freqs_bicross, f, h, w, n_extra=5, device=x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        kwargs = dict()
        dit_blocks = self.pipe.dit.blocks

        for i, block in enumerate(dit_blocks):
            if use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False, **kwargs
                )
                if i == self.start_index - 1:
                    break
            else:
                x = block(x, context, t_mod, freqs, **kwargs)
                if i == self.start_index - 1:
                    break

        vggt_x = x.reshape(x.shape[0], f, h, w, x.shape[-1])
        vggt_x = vggt_x.permute(0, 4, 1, 2, 3)
        patch_token, camera_token, e0 = self.vggt._process_wan_input(
            patch_token=vggt_x, camera_token=camera_token, t=timestep)
        tokens, pos = self.vggt.aggregator._process_aggregator_input(
            patch_token, camera_token)

        B, T, _, _, C = patch_token.shape
        S = T
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for i in range(len(dit_blocks) - self.start_index):
            tokens, frame_idx, frame_intermediates = self.vggt.aggregator._process_frame_attention(
                tokens, B, S, P, C, frame_idx, pos=pos, e0=e0, )
            if i in self.cross_attention_list:
                x, tokens, global_intermediates = self.IRGBlock[i](
                    x_dit=x, x_agg=tokens, context=context, timestep=timestep, t_mod=t_mod,
                    freqs=freqs, freqs_dit=freqs_bi_dit, freqs_agg=freqs_bi_agg,
                    pos=pos, e0=e0, uncond=uncond, **kwargs,
                )
                global_idx += 1
            else:
                if self.pipe.dit.training and use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(dit_blocks[i + self.start_index]),
                        x, context, t_mod, freqs,
                        use_reentrant=False, **kwargs,
                    )
                else:
                    x = dit_blocks[i +
                                   self.start_index](x, context, t_mod, freqs, **kwargs)

                tokens, global_idx, global_intermediates = self.vggt.aggregator._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos, e0=e0)
            for index in range(len(frame_intermediates)):

                concat_inter = torch.cat(
                    [frame_intermediates[index], global_intermediates[index]], dim=-1)
                output_list.append(concat_inter)

        x = self.pipe.dit.head(x, t)
        x = self.pipe.dit.unpatchify(x, (f, h, w))
        if return_prediction:
            prediction = {}
            prediction = self.vggt._head_predction(
                patch_token, self.vggt.aggregator.patch_start_idx, output_list)

            return x, prediction
        else:
            return x, None
