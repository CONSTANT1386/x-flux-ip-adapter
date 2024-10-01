import argparse
import gc
import itertools
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5, load_image_encoder)
from src.flux.modules.layers import IPDoubleStreamBlockProcessor, IPSingleStreamBlockProcessor, ImageProjModel
from src.flux.xflux_pipeline import XFluxSampler
from image_datasets.dataset import ip_dataset_loader, ip_wds_loader

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

class IPAdapter_flux(torch.nn.Module):
    """IP-Adapter for flux"""

    def __init__(self, transformer, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.transformer = transformer
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        print(orig_adapter_sum.item())
        print(orig_ip_proj_sum.item())

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_model_input, timesteps,guidance, pooled_prompt_embeds, encoder_hidden_states, text_ids, latent_image_ids, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)  # image_embeds is [1,768] --> ip_tokens: [1,4,4096] 
        noise_pred = self.transformer(
            img=noisy_model_input,
            txt=encoder_hidden_states,
            y=pooled_prompt_embeds,
            timesteps=timesteps,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=guidance,
            image_proj = ip_tokens,
            )
        return noise_pred

    def load_from_checkpoint(self, ckpt_path:str="", state_dict = None):
        if state_dict is not None:
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        elif ckpt_path != "":
            # Calculate original checksums
            orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
            orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
            print(orig_adapter_sum.item())
            print(orig_ip_proj_sum.item())

            state_dict = torch.load(ckpt_path, map_location="cpu")

            # Load state dict for image_proj_model and adapter_modules
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

            # Calculate new checksums
            new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
            new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
            print(new_adapter_sum.item())
            print(new_ip_proj_sum.item())

            # Verify if the weights have changed
            # assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
            # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

            print(f"Successfully loaded weights from checkpoint {ckpt_path}")
        del state_dict
        gc.collect()

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    image_encoder = load_image_encoder(device)
    return model, vae, t5, clip, image_encoder

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit, vae, t5, clip, image_encoder = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    attn_procs = {}

    if args.double_blocks is None:
        double_blocks_internal = 2
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_internal = 4
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index%double_blocks_internal==0:
            print("setting IP Double Processor for", name)
            attn_procs[name] = IPDoubleStreamBlockProcessor(
              context_dim=4096, hidden_dim=3072
            )
        elif name.startswith("single_blocks") and layer_index%single_blocks_internal==0:
            print("setting IP Single Processor for", name)
            attn_procs[name] = IPSingleStreamBlockProcessor(
              context_dim=4096, hidden_dim=3072
            )
        else:
            attn_procs[name] = attn_processor

    dit.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList([ip_module for ip_module in dit.attn_processors.values() if 'IP' in str(ip_module)])
    image_proj_model = ImageProjModel(
        cross_attention_dim=4096,
        clip_embeddings_dim=768,
        clip_extra_context_tokens=4,
    )
    ip_adapter = IPAdapter_flux(transformer=dit,image_proj_model=image_proj_model,adapter_modules=adapter_modules)

    image_proj_model.requires_grad_(True)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    image_encoder.requires_grad_(False)
    dit.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()
    for param in ip_adapter.adapter_modules.parameters():
        param.requires_grad = True
    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    optimizer_cls = torch.optim.AdamW

    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = optimizer_cls(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = ip_wds_loader(**args.wds_config) if args.use_wds else ip_dataset_loader(**args.local_config)
    train_dataloader_len = args.train_dataloader_len if args.use_wds else len(train_dataloader)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    ip_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ip_adapter, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    # ip_adapter, optimizer, _, lr_scheduler = accelerator.prepare(
    #     ip_adapter, optimizer, deepcopy(train_dataloader), lr_scheduler
    # )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
                999,
                (1024 // 8) * (1024 // 8) // 4,
                shift=True,
            )
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    def load_checkpoint_and_resume():
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split('.')[0]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
                first_epoch = 0
                step_in_epoch = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                # checkpoint = accelerator.load(os.path.join(args.output_dir, path))
                ip_adapter.load_from_checkpoint(ckpt_path=os.path.join(args.output_dir, path))
                checkpoint = torch.load(os.path.join(args.output_dir, path), map_location="cpu")
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Restore global_step, epoch, and step
                initial_global_step = checkpoint['global_step']
                first_epoch = checkpoint['epoch']
                step_in_epoch = checkpoint['step_in_epoch']

        else:
            initial_global_step = 0
            first_epoch = 0
            step_in_epoch = 0
        return initial_global_step, first_epoch, step_in_epoch


    global_step, first_epoch, step_in_epoch = load_checkpoint_and_resume()

    def check_nan(loss):
        if torch.isnan(loss):
            accelerator.print("Loss is NaN. Loading checkpoint and restarting training.")
            return True
        return False

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader, start=step_in_epoch):
            with accelerator.accumulate(ip_adapter):
                img = batch['images']
                prompts = batch['texts']
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)

                with torch.no_grad():
                    image_embeds = image_encoder(
                        batch["clip_images"].to(accelerator.device).to(weight_dtype)).image_embeds
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                torch.cuda.empty_cache()
                # Predict the noise residual and compute loss
                # noise_pred = dit(
                #     img=x_t.to(weight_dtype),
                #     txt=inp['txt'].to(weight_dtype),
                #     y=inp['vec'].to(weight_dtype),
                #     timesteps=t.to(weight_dtype),
                #     img_ids=inp['img_ids'].to(weight_dtype),
                #     txt_ids=inp['txt_ids'].to(weight_dtype),
                #     guidance=guidance_vec.to(weight_dtype),
                #     image_proj = ip_tokens,
                #     )
                model_pred = ip_adapter(noisy_model_input=x_t.to(weight_dtype),
                                latent_image_ids=inp['img_ids'].to(weight_dtype),
                                encoder_hidden_states=inp['txt'].to(weight_dtype),
                                text_ids=inp['txt_ids'].to(weight_dtype),
                                pooled_prompt_embeds=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),
                                image_embeds = image_embeds.to(weight_dtype)
                                )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                if global_step > 1 and check_nan(loss):
                    # Load the checkpoint and restart training
                    global_step, first_epoch, step_in_epoch = load_checkpoint_and_resume()
                    progress_bar.close()
                    progress_bar = tqdm(range(0, args.max_train_steps),
                                        initial=global_step, desc="Steps",
                                        disable=not accelerator.is_local_main_process,)
                    break  # Exit the inner loop to restart training

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # check sum for whether updated
            if global_step % 100 == 0 and accelerator.is_main_process:
                trained_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.image_proj_model.parameters()]))
                trained_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.adapter_modules.parameters()]))
                print(trained_ip_proj_sum.item())
                print(trained_adapter_sum.item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if not args.disable_sampling and global_step % args.sample_every == 0:
                    print(f"Sampling images for step {global_step}...")
                    sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=accelerator.device)
                    images = []
                    for i, prompt in enumerate(args.sample_prompts):
                        result = sampler(prompt=prompt,
                                         width=args.sample_width,
                                         height=args.sample_height,
                                         num_steps=args.sample_steps
                                         )
                        images.append(wandb.Image(result))
                        print(f"Result for prompt #{i} is generated")
                        # result.save(f"{global_step}_prompt_{i}_res.png")
                    wandb.log({f"Results, step {global_step}": images})

            unwrapped_ip_adapter = accelerator.unwrap_model(ip_adapter)
            if global_step >= args.max_train_steps and accelerator.is_local_main_process:
                if (step + 1) % args.gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                save_path = os.path.join(args.output_dir, f"final-checkpoint-{global_step}.safetensors")
                # accelerator.save_state(save_path,model=ip_adapter)
                # torch.save({
                #     'ip_adapter': unwrapped_ip_adapter.adapter_modules.state_dict(),
                #     'image_proj': unwrapped_ip_adapter.image_proj_model.state_dict(),
                # }, save_path)
                # accelerator.print(f"Finally saved state to {save_path}")
                ip_adapter_state_dict = unwrapped_ip_adapter.adapter_modules.state_dict()
                image_proj_state_dict = unwrapped_ip_adapter.image_proj_model.state_dict()
                merged_state_dict = {f'ip_adapter.{k}': v for k, v in ip_adapter_state_dict.items()}
                merged_state_dict.update({f'image_proj.{k}': v for k, v in image_proj_state_dict.items()})
                save_file(merged_state_dict, save_path)
                break

            if global_step % args.checkpointing_steps == 0 and global_step != 0 and accelerator.is_local_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split('.')[0]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        accelerator.print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            os.remove(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.ckpt")
                # accelerator.save_state(save_path)
                # compare whether the weight updated
                torch.save({
                    'ip_adapter': unwrapped_ip_adapter.adapter_modules.state_dict(),
                    'image_proj': unwrapped_ip_adapter.image_proj_model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': lr_scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'step_in_epoch': step + 1
                }, save_path)
                accelerator.print(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        step_in_epoch = 0

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    main()