#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def run_command(cmd, cwd=None):
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code: {result.returncode}")
        sys.exit(result.returncode)
    return result

def main():
    parser = argparse.ArgumentParser(description="Master script to run AgentPoison reproduction and extensions.")
    parser.add_argument("--mode", type=str, choices=["optimize", "evaluate", "all"], default="all",
                        help="Action to perform: optimize trigger, evaluate agent under attack/benign, or both.")
    parser.add_argument("--agent", type=str, choices=["qa", "ehr"], default="qa",
                        help="Target agent: qa (ReAct-StrategyQA) or ehr (EHRAgent).")
    parser.add_argument("--model", type=str, default="dpr",
                        help="Retriever model prefix: dpr, ance, bge, realm.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for optimization and evaluation.")
    parser.add_argument("--adaptive_lambda", type=float, default=0.0,
                        help="Adaptive attacker variance penalty weight (0.0 for standard, >0.0 for adaptive).")
    parser.add_argument("--num_iter", type=int, default=1000,
                        help="Number of iterations for trigger optimization.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation/optimization.")
    parser.add_argument("--backbone", type=str, choices=["gpt", "llama3"], default="gpt",
                        help="Backbone LLM for agent execution.")
    parser.add_argument("--save_root", type=str, default="outputs",
                        help="Directory to save all results and logs.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent / "AgentPoison"
    # Resolve save_root to absolute path to prevent CWD mismatch errors in subprocesses
    args.save_root = str(Path(args.save_root).resolve())
    os.makedirs(args.save_root, exist_ok=True)

    # Define model code
    model_mapping = {
        "dpr": "dpr-ctx_encoder-single-nq-base",
        "realm": "realm-cc-news-pretrained-embedder",
        "ance": "ance-dpr-question-multi",
        "bge": "bge-large-en"
    }
    model_code = model_mapping.get(args.model, args.model)

    # 1. OPTIMIZE MODE
    trigger_dir = Path(args.save_root) / "triggers" / args.agent / f"len10_seed{args.seed}_adapt{args.adaptive_lambda}"
    trigger_dir.mkdir(parents=True, exist_ok=True)
    trigger_path = trigger_dir / "optimized_trigger.json"

    if args.mode in ["optimize", "all"]:
        if trigger_path.exists():
            print(f"Skipping trigger optimization: optimized trigger JSON already exists at {trigger_path}")
        else:
            print(f"\n=== [1/2] Optimizing Trigger for {args.agent} (Seed: {args.seed}, Adaptive: {args.adaptive_lambda}) ===")
            opt_cmd = [
                sys.executable, "algo/trigger_optimization.py",
                "--agent", args.agent,
                "--algo", "ap",
                "--model", model_code,
                "--save_dir", str(trigger_dir),
                "--num_iter", str(args.num_iter),
                "--per_gpu_eval_batch_size", str(args.batch_size),
                "--ppl_filter",
                "--seed", str(args.seed),
                "--adaptive_lambda", str(args.adaptive_lambda)
            ]
            run_command(opt_cmd, cwd=project_root)

    # 2. EVALUATE MODE
    if args.mode in ["evaluate", "all"]:
        print(f"\n=== [2/2] Evaluating Agent {args.agent} under Benign and Attack Conditions ===")
        
        # Verify trigger exists for attack run
        if not trigger_path.exists():
            print(f"Warning: Optimized trigger JSON not found at {trigger_path}. Falling back to default trigger.")
            trigger_arg = []
        else:
            trigger_arg = ["--trigger_path", str(trigger_path)]

        agent_save_dir = Path(args.save_root) / "agent_runs" / args.agent / args.model / f"seed{args.seed}_adapt{args.adaptive_lambda}"
        agent_save_dir.mkdir(parents=True, exist_ok=True)

        if args.agent == "qa":
            # Run benign evaluation
            print("\n--- Running Benign Evaluation ---")
            benign_cmd = [
                sys.executable, "ReAct/run_strategyqa_gpt3.5.py",
                "--model", args.model,
                "--task_type", "benign",
                "--backbone", args.backbone,
                "--save_dir", str(agent_save_dir)
            ]
            run_command(benign_cmd, cwd=project_root)

            # Run adversarial evaluation
            print("\n--- Running Adversarial (Attack) Evaluation ---")
            adv_cmd = [
                sys.executable, "ReAct/run_strategyqa_gpt3.5.py",
                "--model", args.model,
                "--task_type", "adv",
                "--backbone", args.backbone,
                "--save_dir", str(agent_save_dir)
            ] + trigger_arg
            run_command(adv_cmd, cwd=project_root)

            # Evaluate response JSON
            print("\n--- Evaluating Results ---")
            eval_benign_cmd = [sys.executable, "ReAct/eval.py", "-p", str(agent_save_dir / f"{args.model}-ap-benign.jsonl")]
            run_command(eval_benign_cmd, cwd=project_root)

            eval_adv_cmd = [sys.executable, "ReAct/eval.py", "-p", str(agent_save_dir / f"{args.model}-ap-adv.jsonl")]
            run_command(eval_adv_cmd, cwd=project_root)

        elif args.agent == "ehr":
            # Run benign evaluation
            print("\n--- Running Benign Evaluation ---")
            benign_cmd = [
                sys.executable, "EhrAgent/ehragent/main.py",
                "--backbone", args.backbone,
                "--model", args.model,
                "--algo", "ap",
                "--save_dir", str(agent_save_dir)
            ]
            run_command(benign_cmd, cwd=project_root)

            # Run adversarial evaluation
            print("\n--- Running Adversarial (Attack) Evaluation ---")
            adv_cmd = [
                sys.executable, "EhrAgent/ehragent/main.py",
                "--backbone", args.backbone,
                "--model", args.model,
                "--algo", "ap",
                "--attack",
                "--save_dir", str(agent_save_dir)
            ] + trigger_arg
            run_command(adv_cmd, cwd=project_root)

            # Evaluate response JSON
            print("\n--- Evaluating Results ---")
            eval_benign_cmd = [sys.executable, "EhrAgent/ehragent/eval.py", "-p", str(agent_save_dir / args.backbone / f"ap_benign_{args.model}.json")]
            run_command(eval_benign_cmd, cwd=project_root)

            eval_adv_cmd = [sys.executable, "EhrAgent/ehragent/eval.py", "-p", str(agent_save_dir / args.backbone / f"ap_trigger_{args.model}.json")]
            run_command(eval_adv_cmd, cwd=project_root)

if __name__ == "__main__":
    main()
