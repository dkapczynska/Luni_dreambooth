#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
# ]
# ///
"""
DreamBooth fine-tuning for your dog directly from AWS S3.
Author : <you@example.com>
Date   : 2025-07-29
"""

import argparse, os, tempfile, json, shutil, time
from pathlib import Path
from typing import List

import numpy as np

import boto3, torch
from botocore.exceptions import ClientError
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel,
    DDPMScheduler, DiffusionPipeline
)
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger

from dotenv import load_dotenv
load_dotenv()  # this loads variables from .env into environment
import os

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# ---------------------------------------------------------------------------
# 1. S3 DOWNLOAD UTILITIES
# ---------------------------------------------------------------------------
def download_images_from_s3(bucket: str, prefix: str, local_dir: Path) -> List[Path]:
    """
    Download all images under `s3://bucket/prefix/` into local_dir.
    Credentials are taken from the normal AWS search chain
    (env vars, ~/.aws/, IAM role, etc.).
    """
    s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    try:
        # Attempt to list objects in the bucket (lightweight call, shows if access is OK)
        s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
        print(f"Access to bucket {BUCKET_NAME} succeeded!")
    except ClientError as e:
        print(f"Access to bucket {BUCKET_NAME} failed: {e}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="")

    local_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                file_name = Path(key).name
                local_path = local_dir / file_name
                s3.download_file(BUCKET_NAME, key, str(local_path))
                image_paths.append(local_path)
    if not image_paths:
        raise RuntimeError("No images found in the specified S3 location.")
    return image_paths

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def center_crop_and_resize(img_path: Path, size: int = 512) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    # center-crop to square
    min_side = min(im.width, im.height)
    left = (im.width  - min_side) // 2
    top  = (im.height - min_side) // 2
    im = im.crop((left, top, left + min_side, top + min_side))
    return im.resize((size, size), Image.Resampling.LANCZOS)

def save_as_png(img: Image.Image, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest.with_suffix(".png"))

# ---------------------------------------------------------------------------
# 3. TRAINER CLASS
# ---------------------------------------------------------------------------
class DogDreamBooth:
    """
    Fine-tunes Stable Diffusion v1-5 (default) on a small set of
    dog photos using DreamBooth.
    """

    def __init__(
        self,
        instance_dir: Path,
        output_dir: Path,
        instance_token: str,
        class_token: str = "dog",
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        resolution: int = 512,
        lr: float = 5e-6,
        max_steps: int = 800,
        train_text_encoder: bool = False,
        prior_loss_weight: float = 1.0,
        num_class_images: int = 200,
    ):
        self.instance_dir   = instance_dir
        self.output_dir     = output_dir
        self.instance_token = instance_token   # e.g. "sks"
        self.class_token    = class_token      # e.g. "dog"
        self.prompt_inst    = f"a photo of {instance_token} {class_token}"
        self.prompt_class   = f"a photo of a {class_token}"
        self.prior_loss_w   = prior_loss_weight
        self.num_class_imgs = num_class_images
        self.resolution     = resolution
        self.lr             = lr
        self.max_steps      = max_steps
        self.train_text_enc = train_text_encoder
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.accelerator    = Accelerator(mixed_precision="fp16")
        self.logger         = get_logger(__name__, log_level="INFO")

        # Load tokenizer & models
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.text_enc  = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
        self.vae       = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
        self.unet      = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self._freeze()  # keep VAE & (optionally) text encoder frozen

    # ------------------------------------------------------------------ #
    def _freeze(self):
        self.vae.requires_grad_(False)
        if not self.train_text_enc:
            self.text_enc.requires_grad_(False)

    # ------------------------------------------------------------------ #
    def _encode_prompt(self, prompt: str):
        ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_enc(ids)[0]

    # ------------------------------------------------------------------ #
    def _build_dataloader(self):
        """Very small in-memory dataloader for few-shot DreamBooth."""
        from torch.utils.data import Dataset, DataLoader

        class DogSet(Dataset):
            def __init__(self, files: List[Path], size: int):
                self.files = files
                self.size  = size

            def __len__(self): return len(self.files)

            def __getitem__(self, idx):
                img = center_crop_and_resize(self.files[idx], self.size)
                arr = torch.tensor(np.array(img)).permute(2, 0, 1).float()
                arr = arr / 127.5 - 1.0     # scale to [-1, 1]
                return arr

        ds  = DogSet(list(self.instance_dir.glob("*.png")), self.resolution)
        dl  = DataLoader(ds, batch_size=1, shuffle=True)
        return dl

    # ------------------------------------------------------------------ #
    def train(self):
        self.logger.info("🦴 Starting DreamBooth training …")

        dl = self._build_dataloader()

        # Optimizer & LR scheduler
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(
            self.unet.parameters(), lr=self.lr, weight_decay=1e-2
        )
        lr_sched = get_scheduler(
            "constant",
            optimizer=optim,
            num_warmup_steps=0,
            num_training_steps=self.max_steps,
        )

        # Accelerator prepares everything for (multi) GPU training
        self.unet, optim, dl, lr_sched = self.accelerator.prepare(
            self.unet, optim, dl, lr_sched
        )

        global_step = 0
        for epoch in range(1000):          # overshoot, break inside
            for imgs in dl:
                imgs = imgs.to(self.device, dtype=torch.float16)
                with torch.no_grad():
                    latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

                noise      = torch.randn_like(latents)
                timesteps  = torch.randint(0, 1000, (1,), device=self.device)
                noisy_lat  = self.scheduler.add_noise(latents, noise, timesteps)

                # Encode prompt
                enc_hid = self._encode_prompt(self.prompt_inst)

                # UNet forward
                preds = self.unet(noisy_lat, timesteps, enc_hid).sample
                loss  = torch.nn.functional.mse_loss(preds.float(), noise.float(), reduction="mean")

                self.accelerator.backward(loss)
                optim.step(); optim.zero_grad(); lr_sched.step()

                if self.accelerator.is_main_process and global_step % 50 == 0:
                    self.logger.info(f"step {global_step:>4d} | loss={loss.item():.4f}")

                global_step += 1
                if global_step >= self.max_steps:
                    break
            if global_step >= self.max_steps:
                break

        self.accelerator.wait_for_everyone()
        self.save_model()
        self.logger.info("✅ Training complete.")

    # ------------------------------------------------------------------ #
    def save_model(self):
        if self.accelerator.is_main_process:
            self.logger.info("💾 Serializing pipeline …")
            pipe = StableDiffusionPipeline(
                vae       = self.vae.cpu(),
                text_encoder = self.text_enc.cpu(),
                tokenizer = self.tokenizer,
                unet      = self.unet.cpu(),
                scheduler = self.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            pipe.save_pretrained(self.output_dir)
            self.logger.info(f"📂 Model saved to ⇒ {self.output_dir}")
# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DreamBooth from S3 dog photos")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket with dog images")
    p.add_argument("--s3-prefix", required=True,  help="Prefix/folder inside the bucket")
    p.add_argument("--instance-token", default="sks", help="Rare token for your dog")
    p.add_argument("--class-token",    default="dog")
    p.add_argument("--output-dir",     default="dog_model")
    p.add_argument("--steps",          type=int, default=800)
    p.add_argument("--lr",             type=float, default=5e-6)
    return p.parse_args()

def main():
    args = parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="dog_images_"))
    print(f"⏬ Downloading images to {tmpdir}")

    local_paths = download_images_from_s3(args.s3_bucket, args.s3_prefix, tmpdir)

    # Pre-transform all to 512×512 PNG
    for p in tqdm(local_paths, desc="Pre-processing"):
        img = center_crop_and_resize(p)
        save_as_png(img, p.with_suffix(".png"))
        p.unlink()  # remove original

    trainer = DogDreamBooth(
        instance_dir=tmpdir,
        output_dir=Path(args.output_dir),
        instance_token=args.instance_token,
        class_token=args.class_token,
        max_steps=args.steps,
        lr=args.lr,
    )
    trainer.train()

    # Optionally re-upload the final model
    # s3 = boto3.client("s3")
    # s3.upload_file("<local path>", "<bucket>", "<key>")

    shutil.rmtree(tmpdir)  # clean up temp images

if __name__ == "__main__":
    main()
