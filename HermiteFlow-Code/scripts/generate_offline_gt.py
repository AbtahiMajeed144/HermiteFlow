import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import cv2

# Add src to pythonpath so it can resolve models/datasets/etc. before global packages
sys.path.insert(0, os.path.abspath("src"))

from models import create_model
from datasets import create_dataset
from utils.utils import set_seed
from utils.setup import single_setup
from trainers.trainer_hermiteflow import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Offline GT for X4K1000FPS")
    parser.add_argument("-m", "--model-config", type=str, default="configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-idx", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    
    # Overrides matching main.py
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--raft-ckpt", type=str, default=None)
    parser.add_argument("--flowformer-ckpt", type=str, default=None)
    
    args, extra_args = parser.parse_known_args()
    return args, extra_args

def main():
    args, extra_args = parse_args()
    set_seed(args.seed)
    
    # Setup configuration (works for non-ddp scripts)
    config = single_setup(args, extra_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing Dataset with augmentations...")
    # create_dataset returns (train_dataset, val_dataset)
    dataset_trn, _ = create_dataset(config, is_eval=False, logger=None)
    
    # Split dataset
    indices = np.array_split(np.arange(len(dataset_trn)), args.num_splits)[args.split_idx]
    subset = Subset(dataset_trn, indices)
    
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False, # Must be false to properly iterate exactly the split subset once
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Dataset Split {args.split_idx}/{args.num_splits} size: {len(subset)}")
    
    print("Loading Teacher Network (RAFT/FlowFormer)...")
    # model.teacher_flows uses the frozen flow estimator in the base class
    model, _ = create_model(config.arch, ema=False)
    model = model.to(device)
    model.eval()
    
    teacher_raft_iter = int(getattr(config.loss, "teacher_raft_iter", 12))
    
    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)
    split_dir = os.path.join(args.output_dir, f"split_{args.split_idx}")
    os.makedirs(split_dir, exist_ok=True)
    
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating Split {args.split_idx}"):
            # Unpack handles moving to device and extracting endpoints and middles
            # img_xs: (B, 3, 2, H, W), gts: list of K (B, 3, H, W)
            img_xs, gts, t_list = Trainer.unpack(batch, device)
            
            # Predict F_0->t and F_1->t
            target_flows = model.teacher_flows(
                img_xs[:, :, 0], 
                img_xs[:, :, 1], 
                gts, 
                iters=teacher_raft_iter
            )
            
            B = img_xs.shape[0]
            for b in range(B):
                # Save endpoints (convert [0, 1] RGB to [0, 255] BGR for cv2)
                img0 = (img_xs[b, :, 0].permute(1, 2, 0).cpu().numpy() * 255.0)
                img1 = (img_xs[b, :, 1].permute(1, 2, 0).cpu().numpy() * 255.0)
                
                prefix = os.path.join(split_dir, f"sample_{global_idx:06d}")
                cv2.imwrite(f"{prefix}_img0.png", img0[:, :, ::-1].astype(np.uint8))
                cv2.imwrite(f"{prefix}_img1.png", img1[:, :, ::-1].astype(np.uint8))
                
                for k in range(len(gts)):
                    imgt = (gts[k][b].permute(1, 2, 0).cpu().numpy() * 255.0)
                    cv2.imwrite(f"{prefix}_imgt_{k}.png", imgt[:, :, ::-1].astype(np.uint8))
                    
                    # Convert float32 flow to float16
                    flow_0_t = target_flows[k][0][b].cpu().numpy().astype(np.float16)
                    flow_1_t = target_flows[k][1][b].cpu().numpy().astype(np.float16)
                    
                    t_val = t_list[k][b].cpu().item()
                    
                    np.savez_compressed(
                        f"{prefix}_flow_{k}.npz",
                        flow_0_t=flow_0_t,
                        flow_1_t=flow_1_t,
                        t=t_val
                    )
                
                global_idx += 1

if __name__ == "__main__":
    main()
