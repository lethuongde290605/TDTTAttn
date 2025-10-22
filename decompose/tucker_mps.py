"""
Tucker-MPS compression for OPT models
Similar to eigen_attn.py but using Tucker-MPS decomposition
"""
import torch
import torch.nn as nn
import copy
import gc

from models.decompose_modules import OPTTuckerMPSDecoderLayer


def tucker_mps_compress(
    lm,
    args,
    dataloader,
    logger=None,
    mps_eps=0.99,
    hooi_ranks=None,
):
    """
    Apply Tucker-MPS compression to OPT model layers.
    
    Args:
        lm: Language model wrapper
        args: Arguments object
        dataloader: Calibration data loader
        logger: Logger object
        mps_eps: MPS decomposition epsilon threshold
        hooi_ranks: HOOI decomposition ranks
    """
    if logger:
        logger.info("Starting Tucker-MPS compression...")
    else:
        print("Starting Tucker-MPS compression...")
    
    # Only support OPT for now
    assert('opt' in args.net.lower()), "Currently only support OPT model"
    
    if hooi_ranks is None:
        hooi_ranks = [6, 6, 8]
    
    # Setup model
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # Get decoder layers
    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # Catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    # Collect calibration data
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # Move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    # Get attention mask
    attention_mask = cache["attention_mask"]

    # Compress each layer
    for i in range(len(layers)):
        if logger:
            logger.info(f"Compressing layer {i}/{len(layers)}...")
        else:
            print(f"Compressing layer {i}/{len(layers)}...")
        
        layer = layers[i].to(dev)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Get original layer output
                output_hr = torch.stack([
                    layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] 
                    for j in range(args.nsamples)
                ])
                
                # Create compressed layer
                class Args:
                    pass
                layer_args = Args()
                
                compressed_layer = OPTTuckerMPSDecoderLayer(
                    ori_layer=layer,
                    args=layer_args,
                    config=lm.model.config,
                    mps_eps=mps_eps,
                    hooi_ranks=hooi_ranks,
                ).to(dev)
                
                # Get compressed layer output
                output_lr = torch.stack([
                    compressed_layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] 
                    for j in range(args.nsamples)
                ])
                
                # Calculate error
                error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr)).item()
                
                # Calculate compression ratio
                original_params = sum(p.numel() for p in layer.self_attn.parameters())
                compressed_params = sum(p.numel() for p in compressed_layer.self_attn.parameters())
                compression_ratio = compressed_params / original_params
                
                if logger:
                    logger.info(f"Layer {i}: error={error:.6f}, "
                              f"compression_ratio={compression_ratio:.4f}, "
                              f"space_saved={((1-compression_ratio)*100):.2f}%, "
                              f"max_memory={torch.cuda.max_memory_allocated(lm._device) / 1024**2:.2f}MB")
                else:
                    print(f"Layer {i}: error={error:.6f}, "
                          f"compression_ratio={compression_ratio:.4f}, "
                          f"space_saved={((1-compression_ratio)*100):.2f}%")
                
                # Update inputs for next layer
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for j in range(args.nsamples):
                            inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        # Replace layer
        compressed_layer.half()
        layers[i] = compressed_layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache
    
    if logger:
        logger.info("Tucker-MPS compression completed!")
    else:
        print("Tucker-MPS compression completed!")
    
    return model
