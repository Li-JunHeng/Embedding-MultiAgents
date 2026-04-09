import argparse
import json
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from methods.slot_mas import SlotMASMethod
from methods.memory_mas import MemoryMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "slot_mas", "memory_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', 'latent_mas', 'slot_mas', or 'memory_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or local path (e.g. 'Qwen/Qwen3-8B' or a local checkpoint directory).")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--num_slots", type=int, default=4, help="Number of slots for slot_mas compression")
    parser.add_argument("--slot_dim", type=int, default=64, help="Slot dimension for slot_mas bottleneck (default 64)")
    parser.add_argument("--compressor_path", type=str, default=None, help="Path to trained compressor weights for slot_mas")
    parser.add_argument("--memory_dim", type=int, default=256, help="Compressed retrieval dimension for memory_mas")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to trained LatentMemoryAdapter weights for memory_mas")
    parser.add_argument("--memory_segment_length", type=int, default=4, help="Segment length for hierarchical shared memory in memory_mas")
    parser.add_argument("--memory_top_agents", type=int, default=2, help="Top agent summaries to route through in memory_mas")
    parser.add_argument("--memory_top_clusters", type=int, default=4, help="Top thought clusters to retrieve in memory_mas")
    parser.add_argument("--memory_top_segments", type=int, default=4, help="Top L1 segments to fetch in memory_mas")
    parser.add_argument("--memory_max_prefix_tokens", type=int, default=64, help="Maximum retrieved prefix tokens injected into each agent in memory_mas")
    parser.add_argument("--memory_gate_scale", type=float, default=4.0, help="Sigmoid scale applied to cosine gating in memory_mas")
    parser.add_argument("--memory_merge_threshold", type=float, default=0.92, help="Cosine threshold for merging segments into consensus clusters in memory_mas")
    parser.add_argument("--memory_difference_threshold", type=float, default=0.55, help="Cosine threshold below which segments become difference clusters in memory_mas")
    parser.add_argument("--memory_difference_boost", type=float, default=1.25, help="Priority boost for difference clusters in memory_mas")
    parser.add_argument("--memory_consensus_penalty", type=float, default=0.85, help="Priority downweight for consensus clusters in memory_mas")
    parser.add_argument(
        "--memory_device",
        type=str,
        default=None,
        help="For memory_mas only: device for LatentMemoryAdapter and PerSampleMemoryBank tensors. "
        "Default: same as --device. Example: --device cuda:0 --memory_device cuda:1 puts Qwen on GPU0 "
        "and memory compression/retrieval on GPU1.",
    )
    parser.add_argument("--allow_cpu_offload", action="store_true", help="Allow HuggingFace auto/offloaded placement onto CPU or disk. Disabled by default because it can be extremely slow.")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    if args.method == "slot_mas":
        args.use_vllm = False
    if args.method == "memory_mas" and args.use_vllm:
        raise ValueError("memory_mas currently supports only the HuggingFace backend. Please rerun without --use_vllm.")
    
    set_seed(args.seed)
    device = auto_device(args.device)
    if args.method == "memory_mas" and args.memory_device is not None:
        args.memory_device = str(torch.device(args.memory_device))
    elif args.method == "memory_mas":
        args.memory_device = str(device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == 'slot_mas':
        method = SlotMASMethod(
            model,
            latent_steps=args.latent_steps,
            num_slots=args.num_slots,
            slot_dim=args.slot_dim,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'memory_mas':
        method = MemoryMASMethod(
            model,
            latent_steps=args.latent_steps,
            memory_dim=args.memory_dim,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    
    # Load results in JSON format
    result_dict = {
        "method": args.method,
        "model": args.model_name,
        "task": args.task,
        "prompt": args.prompt,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total": args.max_samples,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4),
    }
    if args.method == "slot_mas":
        result_dict["num_slots"] = args.num_slots
        result_dict["slot_dim"] = args.slot_dim
        result_dict["latent_steps"] = args.latent_steps
        result_dict["msg_bytes_per_agent"] = method._msg_bytes
        result_dict["total_msg_bytes"] = method._msg_bytes * 3  # 3 non-judger agents
    elif args.method == "latent_mas":
        result_dict["latent_steps"] = args.latent_steps
    elif args.method == "memory_mas":
        result_dict["latent_steps"] = args.latent_steps
        result_dict["memory_dim"] = args.memory_dim
        result_dict["adapter_path"] = args.adapter_path
        result_dict["memory_segment_length"] = args.memory_segment_length
        result_dict["memory_top_agents"] = args.memory_top_agents
        result_dict["memory_top_clusters"] = args.memory_top_clusters
        result_dict["memory_top_segments"] = args.memory_top_segments
        result_dict["memory_max_prefix_tokens"] = args.memory_max_prefix_tokens
        result_dict["memory_gate_scale"] = args.memory_gate_scale
        result_dict["memory_merge_threshold"] = args.memory_merge_threshold
        result_dict["memory_difference_threshold"] = args.memory_difference_threshold
        result_dict["memory_difference_boost"] = args.memory_difference_boost
        result_dict["memory_consensus_penalty"] = args.memory_consensus_penalty
    print(json.dumps(result_dict, ensure_ascii=False))



if __name__ == "__main__":
    main()
