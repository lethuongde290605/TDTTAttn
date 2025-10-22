"""
Test script for Tucker-MPS compression on OPT model
"""
import torch
import torch.nn as nn
from transformers import OPTForCausalLM, OPTConfig
from models.decompose_modules import OPTTuckerMPSDecoderLayer
import argparse


def test_tucker_mps_compression():
    """Test Tucker-MPS compression on a single OPT layer"""
    
    print("="*80)
    print("Testing Tucker-MPS Compression on OPT Model")
    print("="*80)
    
    # Load a small OPT model for testing
    model_name = "facebook/opt-125m"  # Small model for quick testing
    print(f"\nLoading model: {model_name}")
    
    model = OPTForCausalLM.from_pretrained(model_name)
    config = model.config
    
    print(f"Model config:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    
    # Get first decoder layer for testing
    original_layer = model.model.decoder.layers[0]
    
    print(f"\n{'='*80}")
    print("Original Layer Weights:")
    print(f"{'='*80}")
    print(f"K projection weight shape: {original_layer.self_attn.k_proj.weight.shape}")
    print(f"Q projection weight shape: {original_layer.self_attn.q_proj.weight.shape}")
    print(f"V projection weight shape: {original_layer.self_attn.v_proj.weight.shape}")
    print(f"Output projection weight shape: {original_layer.self_attn.out_proj.weight.shape}")
    
    # Create args object (minimal)
    class Args:
        def __init__(self):
            pass
    
    args = Args()
    
    # Test with different compression parameters
    mps_eps_values = [0.99, 0.95, 0.90]
    hooi_ranks_values = [[6, 6, 8], [5, 5, 6], [4, 4, 5]]
    
    for mps_eps, hooi_ranks in zip(mps_eps_values, hooi_ranks_values):
        print(f"\n{'='*80}")
        print(f"Testing with MPS eps={mps_eps}, HOOI ranks={hooi_ranks}")
        print(f"{'='*80}")
        
        try:
            # Create compressed layer
            compressed_layer = OPTTuckerMPSDecoderLayer(
                ori_layer=original_layer,
                args=args,
                config=config,
                mps_eps=mps_eps,
                hooi_ranks=hooi_ranks
            )
            
            print(f"\n✓ Successfully created compressed layer!")
            
            # Test forward pass
            batch_size = 2
            seq_len = 10
            hidden_size = config.hidden_size
            
            dummy_input = torch.randn(batch_size, seq_len, hidden_size)
            
            print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
            
            # Original layer forward
            with torch.no_grad():
                original_output = original_layer(dummy_input)[0]
                print(f"Original output shape: {original_output.shape}")
                
                # Compressed layer forward
                compressed_output = compressed_layer(dummy_input)[0]
                print(f"Compressed output shape: {compressed_output.shape}")
                
                # Calculate reconstruction error
                error = torch.norm(original_output - compressed_output) / torch.norm(original_output)
                print(f"\n📊 Reconstruction error: {error.item():.6f}")
                
                # Calculate compression ratio
                original_params = sum(p.numel() for p in original_layer.self_attn.parameters())
                compressed_params = sum(p.numel() for p in compressed_layer.self_attn.parameters())
                compression_ratio = compressed_params / original_params
                
                print(f"📦 Compression statistics:")
                print(f"   - Original parameters: {original_params:,}")
                print(f"   - Compressed parameters: {compressed_params:,}")
                print(f"   - Compression ratio: {compression_ratio:.4f}")
                print(f"   - Space saved: {(1 - compression_ratio) * 100:.2f}%")
                
        except Exception as e:
            print(f"\n❌ Error during compression: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")


def test_compression_only():
    """Test only the compression function without full layer"""
    from decompose.tucker_mps_utils import combined_mps_hooi_compression
    
    print("="*80)
    print("Testing Tucker-MPS Compression Function Only")
    print("="*80)
    
    # Test with 768x768 matrix (OPT-125m size)
    weight_matrix = torch.randn(768, 768)
    print(f"\nOriginal weight matrix shape: {weight_matrix.shape}")
    
    try:
        compressed = combined_mps_hooi_compression(
            weight_matrix,
            mps_eps=0.99,
            hooi_ranks=[6, 6, 8],
            verbose=True
        )
        print(f"\n✓ Compression successful!")
        print(f"Compressed shape: {compressed.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tucker-MPS compression")
    parser.add_argument("--test-type", type=str, default="full", 
                       choices=["full", "compression-only"],
                       help="Type of test to run")
    
    args = parser.parse_args()
    
    if args.test_type == "compression-only":
        test_compression_only()
    else:
        test_tucker_mps_compression()
