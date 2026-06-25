import sys
import os
import logging
import torch
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import create_model
from src.utils.profiler import Profiler

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify")

def verify_model(config_path):
    print(f"\n{'='*50}")
    print(f"Verifying {config_path}")
    print(f"{'='*50}")
    
    # 1. Load config
    config = OmegaConf.load(config_path)
    
    # We only care about the arch config for model creation
    from src.models.hermite_vfi.configs import HermiteFlowConfig
    arch_config = HermiteFlowConfig.create(config.arch)
    
    # 2. Create Profiler
    profiler = Profiler(logger)
    
    # 3. Create Model
    # This will trigger __init__, which calls _load_and_freeze_decoder
    model, _ = create_model(arch_config, ema=False)
    
    # 4. Print parameter breakdown
    profiler.get_module_breakdown(model)

    # 5. Run dummy tensor pass with precomputed_flows
    print(f"\nRunning dummy forward pass with precomputed flows...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    B, C, H, W = 2, 3, 256, 256
    img_xs = torch.rand(B, 3, 2, H, W, device=device) # RGB images
    t = [torch.full((B,), 0.5, device=device)]
    coord_inputs = [(torch.zeros(B, 1, H, W, 3, device=device), None)]
    
    # Precomputed flows (B, 2, 2, H, W) -> [f01, f10]
    precomputed_flows = torch.rand(B, 2, 2, H, W, device=device)
    
    with torch.no_grad():
        try:
            outputs = model(
                img_xs, 
                coord=coord_inputs, 
                t=t, 
                precomputed_flows=precomputed_flows
            )
            print("Successfully executed forward pass with precomputed flows!")
            print(f"Output imgt_pred shape: {outputs['imgt_pred'][0].shape}")
        except Exception as e:
            print(f"Failed forward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_model("configs/hermiteflow/hermiteflow_r.yaml")
    verify_model("configs/hermiteflow/hermiteflow_f.yaml")
