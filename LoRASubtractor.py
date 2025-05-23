from safetensors.torch import load_file, save_file
import torch

def subtract_safetensors(file1_path, file2_path, output_path):
    # Load the two safetensor files
    tensor1 = load_file(file1_path)
    tensor2 = load_file(file2_path)
    
    # Check if the tensors have the same keys
    if tensor1.keys() != tensor2.keys():
        raise ValueError("Tensor files have different keys. They must have identical structures.")
    
    # Subtract tensors
    result = {}
    for key in tensor1.keys():
        if tensor1[key].shape != tensor2[key].shape:
            raise ValueError(f"Shape mismatch for key {key}: {tensor1[key].shape} vs {tensor2[key].shape}")
        result[key] = tensor1[key] - tensor2[key]
    
    # Save the result to a new safetensor file
    save_file(result, output_path)
    print(f"Result saved to {output_path}")

# Example usage
if __name__ == "__main__":
    file1_path = "./main.safetensors"
    file2_path = "./subtract.safetensors"
    output_path = "./output.safetensors"
    
    try:
        subtract_safetensors(file1_path, file2_path, output_path)
    except Exception as e:
        print(f"Error: {e}")