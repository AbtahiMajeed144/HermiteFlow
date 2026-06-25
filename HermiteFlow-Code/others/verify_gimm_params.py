import torch
from collections import defaultdict

def analyze_gimm_checkpoint(ckpt_path):
    print(f"\n{'='*60}")
    print(f"Analyzing Original GIMM-VFI Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
    module_params = defaultdict(int)
    total_params = 0
    
    for key, tensor in state_dict.items():
        # Remove "module." prefix if it exists
        clean_key = key.replace("module.", "")
        
        # Extract the top-level module name (e.g., "flow_estimator.model.weight" -> "flow_estimator")
        parts = clean_key.split(".")
        top_module = parts[0]
        
        num_params = tensor.numel()
        module_params[top_module] += num_params
        total_params += num_params
        
    print("\n[MODULE BREAKDOWN]")
    for module_name, count in sorted(module_params.items()):
        print(f"  {module_name:<25} {count/1e6:>8.4f}M")
        
    print("-" * 40)
    print(f"  {'TOTAL:':<25} {total_params/1e6:>8.4f}M\n")

if __name__ == "__main__":
    analyze_gimm_checkpoint("pretrained/gimmvfi_r_arb.pt")
    analyze_gimm_checkpoint("pretrained/gimmvfi_f_arb.pt")
