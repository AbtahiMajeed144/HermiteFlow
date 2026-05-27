import os
import glob
import math
import torch
import numpy as np
import PIL.Image
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model.pwcnet as pwcnet
from model.m2m import hermite_flow_scale

def compute_epe(flow_pred, flow_gt):
    """
    Computes End-Point Error between two flow fields.
    flow shape: (1, 2, H, W)
    Returns scalar mean EPE.
    """
    epe = torch.norm(flow_pred - flow_gt, p=2, dim=1).mean()
    return epe.item()

def run_epe_experiment(dataset_path):
    print("Loading PWC-Net...")
    netFlow = pwcnet.Network().cuda().eval()
    
    # We don't load specific weights for PWC-Net here because M2M's pwcnet 
    # uses the weights from the loaded M2M model or pre-trained pwcnet.
    # We will try to load M2M's weights to ensure pwcnet is initialized correctly.
    # The prompt says "using the existing flow estimator". We will load the full model weights
    # and extract the pwcnet part, or just load model.pkl if it exists.
    if os.path.exists('./model.pkl'):
        state_dict = torch.load('./model.pkl')
        # Filter pwcnet weights
        pwc_dict = {k.replace('netFlow.', ''): v for k, v in state_dict.items() if 'netFlow' in k}
        netFlow.load_state_dict(pwc_dict)
        print("Loaded PWC-Net weights from M2M model.pkl")
    else:
        print("WARNING: model.pkl not found, using uninitialized PWC-Net weights.")

    # Assume standard triplet structure for SNU-FILM:
    # dataset_path/00000/0.png, 1.png, 2.png
    # Where 0 and 2 are inputs, 1 is the intermediate frame at t=0.5
    sequences = sorted(glob.glob(os.path.join(dataset_path, '*')))
    
    if not sequences:
        print(f"No sequences found at {dataset_path}. Please ensure it follows the expected structure (subfolders with 0.png, 1.png, 2.png).")
        return

    epe_linear_list = []
    epe_hermite_list = []

    print(f"Starting evaluation on {len(sequences)} sequences...")
    for seq in tqdm(sequences):
        img0_path = os.path.join(seq, '0.png')
        img1_path = os.path.join(seq, '2.png')
        imgt_path = os.path.join(seq, '1.png')

        if not (os.path.exists(img0_path) and os.path.exists(img1_path) and os.path.exists(imgt_path)):
            continue

        # Load images
        im0 = np.array(PIL.Image.open(img0_path))[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        im1 = np.array(PIL.Image.open(img1_path))[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        imt = np.array(PIL.Image.open(imgt_path))[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)

        im0 = torch.FloatTensor(np.ascontiguousarray(im0.transpose(2, 0, 1)[None, :, :, :])).cuda()
        im1 = torch.FloatTensor(np.ascontiguousarray(im1.transpose(2, 0, 1)[None, :, :, :])).cuda()
        imt = torch.FloatTensor(np.ascontiguousarray(imt.transpose(2, 0, 1)[None, :, :, :])).cuda()

        # Target time for middle frame is 0.5
        t = torch.tensor([0.5]).view(1, 1, 1, 1).cuda()

        with torch.no_grad():
            # 1. Compute bidirectional flows between inputs
            F_0to1, F_1to0 = netFlow.bidir(im0, im1)

            # 2. Compute pseudo-ground-truth flow from I0 to It
            F_0tot_gt, _ = netFlow.bidir(im0, imt)

            # 3. Compute intermediate flow with linear model
            F_t0_linear = F_0to1 * t
            
            # 4. Compute intermediate flow with hermite model
            F_t0_hermite, _ = hermite_flow_scale(F_0to1, F_1to0, t)

            # 5. Compute EPE
            epe_lin = compute_epe(F_t0_linear, F_0tot_gt)
            epe_her = compute_epe(F_t0_hermite, F_0tot_gt)

        epe_linear_list.append(epe_lin)
        epe_hermite_list.append(epe_her)

    mean_epe_lin = np.mean(epe_linear_list)
    mean_epe_her = np.mean(epe_hermite_list)

    print("\n--- EPE Results ---")
    print(f"Linear Mean EPE:  {mean_epe_lin:.4f}")
    print(f"Hermite Mean EPE: {mean_epe_her:.4f}")
    
    wins = sum(1 for l, h in zip(epe_linear_list, epe_hermite_list) if h < l)
    losses = sum(1 for l, h in zip(epe_linear_list, epe_hermite_list) if h > l)
    ties = len(epe_linear_list) - wins - losses
    
    print(f"\nHermite vs Linear:")
    print(f"Wins:   {wins} sequences")
    print(f"Losses: {losses} sequences")
    print(f"Ties:   {ties} sequences")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Isolated EPE experiment for SNU-FILM.")
    parser.add_argument('--dataset', type=str, default='./SNU-FILM-arb-Hard', help="Path to SNU-FILM split")
    args = parser.parse_args()
    
    run_epe_experiment(args.dataset)
