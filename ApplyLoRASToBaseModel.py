import torch
from safetensors.torch import load_file, save_file

def ensure_float32(weights):
    return {
        k: v.to(torch.float32) if str(v.dtype).startswith("torch.float8") else v
        for k, v in weights.items()
    }

def merge_lora_into_base(base_weights, lora_weights, alpha=1.0, lora_prefix="diffusion_model."):
    merged = base_weights.copy()

    for key in lora_weights:
        if key.endswith(".lora_A.weight") and key.startswith(lora_prefix):
            # Strip prefix and replace suffix
            lora_base = key[len(lora_prefix):-len(".lora_A.weight")]
            base_key = f"{lora_base}.weight"
            A_key = key
            B_key = f"{lora_prefix}{lora_base}.lora_B.weight"

            if base_key in merged:
                A = lora_weights[A_key]
                B = lora_weights[B_key]
                delta = torch.matmul(B, A) * alpha
                print(f"✅ Applying LoRA to: {base_key}")
                merged[base_key] = merged[base_key] + delta.to(merged[base_key].dtype)
            else:
                print(f"⚠️ Base key not found: {base_key}")
    return merged

# === File paths ===
base_path = "./wan2.1_t2v_14B_fp8_scaled.safetensors"
lora1_path = "./BigBoobs_e20.safetensors"
lora2_path = "./SmallBoobs_e20.safetensors"
output_path = "./merged_model.safetensors"

# === Load weights ===
print("Loading base model...")
base_weights = ensure_float32(load_file(base_path))

print("Applying first LoRA...")
lora1 = load_file(lora1_path)
merged_weights = merge_lora_into_base(base_weights, lora1, alpha=1.0)

print("Applying second LoRA...")
lora2 = load_file(lora2_path)
merged_weights = merge_lora_into_base(merged_weights, lora2, alpha=-1.0)

# === Save result ===
print(f"Saving merged model to: {output_path}")
save_file(merged_weights, output_path)

print("✅ Done. Merged model saved.")
