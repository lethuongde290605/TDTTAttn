"""
Main script for Tucker-MPS compression with full evaluation
Similar to main_eigen_attn.py but using Tucker-MPS compression
"""
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig

from models.LMClass import LMClass
from datautils import get_loaders
from decompose.tucker_mps import tucker_mps_compress
import logging
from pprint import pprint

def setup_logger(args):
    """Setup logger"""
    log_dir = f"./log/{args.model_family}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/tucker_mps_eps{args.mps_eps}_ranks{args.hooi_ranks}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def evaluate_ppl(lm, args, logger):
    """Evaluate perplexity on WikiText2 and C4"""
    results = {}
    
    for dataset in ["wikitext2", "c4"]:
        logger.info(f"Evaluating perplexity on {dataset}...")
        
        cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            logger.info(f"Loaded test data from {cache_testloader}")
        else:
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(testloader, cache_testloader)
        
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // lm.seqlen
        use_cache = lm.model.config.use_cache
        lm.model.config.use_cache = False
        lm.model.eval()
        
        nlls = []
        for i in tqdm(range(nsamples), desc=f"Evaluating {dataset}"):
            batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
            
            if "opt" in args.net.lower():
                outputs = lm.model.model.decoder(batch)
            elif "llama" in args.net.lower():
                outputs = lm.model.model(batch)
            elif "mpt" in args.net.lower():
                outputs = lm.model.transformer(batch)
            else:
                raise ValueError(f"Model {args.net} not supported")
            
            hidden_states = outputs[0]
            logits = lm.model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][:, 1:].to(lm.device)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * lm.seqlen
            nlls.append(neg_log_likelihood)
            
            if args.limit > 0 and i >= args.limit:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
        logger.info(f'{dataset} perplexity: {ppl.item():.4f}')
        lm.model.config.use_cache = use_cache
        results[dataset] = ppl.item()
    
    return results


@torch.no_grad()
def evaluate_tasks(lm, args, logger):
    """Evaluate on lm-eval tasks"""
    if args.tasks == "":
        return {}
    
    logger.info(f"Evaluating on tasks: {args.tasks}")
    
    import lm_eval
    task_manager = lm_eval.tasks.TaskManager()
    
    t_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
        limit=None if args.limit == -1 else args.limit,
    )
    
    logger.info("Task results:")
    pprint(t_results)
    
    return t_results


def calculate_compression_stats(model, args, logger):
    """Calculate and log compression statistics"""
    if "opt" in args.net.lower():
        layers = model.model.decoder.layers
    elif "llama" in args.net.lower():
        layers = model.layers
    elif "mpt" in args.net.lower():
        layers = model.transformer.blocks
    else:
        return
    
    total_original = 0
    total_compressed = 0
    
    for i, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # Count parameters
            if hasattr(attn, 'compressed_dim_k'):
                # Tucker-MPS compressed layer
                compressed_params = sum(p.numel() for p in attn.parameters())
                # Estimate original size (assuming standard dimensions)
                embed_dim = attn.embed_dim
                original_params = 4 * embed_dim * embed_dim  # Q, K, V, O projections
                
                total_compressed += compressed_params
                total_original += original_params
                
                logger.info(f"Layer {i}: {compressed_params:,} params (compressed from {original_params:,})")
    
    if total_original > 0:
        compression_ratio = total_compressed / total_original
        space_saved = (1 - compression_ratio) * 100
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Overall Compression Statistics:")
        logger.info(f"  Original attention parameters: {total_original:,}")
        logger.info(f"  Compressed attention parameters: {total_compressed:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.4f}")
        logger.info(f"  Space saved: {space_saved:.2f}%")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                       help="Model name or path")
    parser.add_argument("--net", type=str, default="opt-125m",
                       help="Network name for logging")
    parser.add_argument("--model_family", type=str, default="opt",
                       help="Model family (opt, llama, mpt)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory for datasets")
    parser.add_argument("--attn_implementation", type=str, default="eager",
                       help="Attention implementation (eager, sdpa, flash_attention_2)")
    parser.add_argument("--load_low_rank", action="store_true",
                       help="Load pre-compressed low-rank model")
    parser.add_argument("--save_dir", type=str, default="./compressed_models",
                       help="Directory to save compressed model")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save evaluation outputs")
    
    # Tucker-MPS compression arguments
    parser.add_argument("--mps_eps", type=float, default=0.99,
                       help="MPS decomposition epsilon threshold (0.9-0.99)")
    parser.add_argument("--hooi_ranks", type=int, nargs=3, default=[6, 6, 8],
                       help="HOOI decomposition ranks [r1, r2, r3]")
    parser.add_argument("--hooi_iter", type=int, default=100,
                       help="Max iterations for HOOI")
    parser.add_argument("--hooi_tol", type=float, default=1e-7,
                       help="Convergence tolerance for HOOI")
    
    # Calibration data arguments
    parser.add_argument("--nsamples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                       help="Calibration dataset (wikitext2, c4)")
    
    # Evaluation arguments
    parser.add_argument("--eval_ppl", action="store_true",
                       help="Evaluate perplexity on WikiText2 and C4")
    parser.add_argument("--tasks", type=str, default="",
                       help="Evaluation tasks (comma-separated)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                       help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=-1,
                       help="Limit number of evaluation samples")
    
    # Device arguments
    parser.add_argument("--multigpu", action="store_true",
                       help="Use multiple GPUs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    
    # Baseline evaluation
    parser.add_argument("--evaluate_baseline", action="store_true",
                       help="Only evaluate baseline model without compression")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args)
    
    logger.info("="*80)
    logger.info("Tucker-MPS Compression for Language Models")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"MPS epsilon: {args.mps_eps}")
    logger.info(f"HOOI ranks: {args.hooi_ranks}")
    logger.info(f"Calibration samples: {args.nsamples}")
    logger.info("="*80)
    
    # Load model
    logger.info("Loading model...")
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    
    # Evaluate baseline if requested
    if args.evaluate_baseline:
        logger.info("\n" + "="*80)
        logger.info("Evaluating BASELINE model (no compression)")
        logger.info("="*80)
        
        results = {}
        if args.eval_ppl:
            ppl_results = evaluate_ppl(lm, args, logger)
            results.update(ppl_results)
        
        if args.tasks:
            task_results = evaluate_tasks(lm, args, logger)
            results.update(task_results)
        
        logger.info("\nBaseline Results:")
        pprint(results)
        return
    
    # Load calibration data
    logger.info(f"\nLoading calibration data from {args.calib_dataset}...")
    dataloader, _ = get_loaders(
        args.calib_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=lm.seqlen,
        nsamples=args.nsamples,
    )
    
    # Apply Tucker-MPS compression
    logger.info("\n" + "="*80)
    logger.info("Applying Tucker-MPS Compression")
    logger.info("="*80)
    
    compressed_model = tucker_mps_compress(
        lm=lm,
        args=args,
        dataloader=dataloader,
        logger=logger,
        mps_eps=args.mps_eps,
        hooi_ranks=args.hooi_ranks,
    )
    
    # Calculate compression statistics
    calculate_compression_stats(compressed_model, args, logger)
    
    # Move model back to device for evaluation
    logger.info(f"\nMoving compressed model to {lm.device}...")
    if "opt" in args.net.lower():
        lm.model = lm.model.to(lm.device)
    elif "llama" in args.net.lower():
        lm.model = lm.model.to(lm.device)
    elif "mpt" in args.net.lower():
        lm.model = lm.model.to(lm.device)
    
    # Evaluate compressed model
    logger.info("\n" + "="*80)
    logger.info("Evaluating COMPRESSED model")
    logger.info("="*80)
    
    results = {}
    
    if args.eval_ppl:
        ppl_results = evaluate_ppl(lm, args, logger)
        results.update(ppl_results)
        print(f"\n[DEBUG] PPL evaluation completed, results: {results}", flush=True)
    
    if args.tasks:
        task_results = evaluate_tasks(lm, args, logger)
        results.update(task_results)
    
    print(f"\n[DEBUG] About to print final results", flush=True)
    logger.info("\n" + "="*80)
    logger.info("Final Results:")
    logger.info("="*80)
    pprint(results)
    
    # Save compressed model
    print(f"\n[DEBUG] Preparing to save model to {args.save_dir}", flush=True)
    save_path = os.path.join(args.save_dir, f"{args.net}_tucker_mps_eps{args.mps_eps}")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"\nSaving compressed model to {save_path}...")
    
    print(f"[DEBUG] Calling torch.save...", flush=True)
    torch.save(lm.model.state_dict(), f"{save_path}/model.pt")
    print(f"[DEBUG] Model saved successfully", flush=True)
    
    # Save results to output directory
    import json
    print(f"[DEBUG] Saving results JSON...", flush=True)
    results_file = os.path.join(args.output_dir, f"{args.net}_tucker_mps_eps{args.mps_eps}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    print(f"[DEBUG] JSON saved successfully", flush=True)
    
    logger.info("Done!")
    print(f"\n[DEBUG] Script completed successfully!", flush=True)


if __name__ == "__main__":
    main()
