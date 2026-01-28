"""
Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?

Usage:
    python run_attack.py --model_path /path/to/model --device cuda:0
"""

import gc
import time
import argparse
import pandas as pd
import torch
import numpy as np
from torch import optim
import json
import os
from datetime import datetime

from llm_attacks.minimal_gcg.opt_utils import get_logits, target_loss, load_model_and_tokenizer
from llm_attacks import get_nonascii_toks
from llm_attacks import get_embedding_matrix, get_embeddings, get_embedding_layer
from mask_gcg_utils import (
    token_gradients,
    sample_control,
    change_lambda_and_lr,
    SuffixManager,
    load_conversation_template,
    get_filtered_cands,
    MaskMLP,
    initialize_attention_guided_mask,
    update_mask_with_attention_guidance,
    smart_pruning_strategy
)
from config import *


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    """Generate model output"""
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attn_masks,
        generation_config=gen_config,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    )[0]

    return output_ids[assistant_role_slice.stop:]


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    """Check if attack succeeded"""
    gen_str = tokenizer.decode(generate(
        model, tokenizer, input_ids, assistant_role_slice, gen_config=gen_config
    )).strip()
    jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
    return gen_str, jailbroken


def run_single_attack(
    model, 
    tokenizer, 
    conv_template,
    user_prompt, 
    target, 
    device,
    adv_string_init=None,
    max_steps=None,
    verbose=True
):
    """
    Run a single Mask-GCG attack
    """
    if max_steps is None:
        max_steps = NUM_STEPS
    
    if adv_string_init is None:
        adv_string_init = ADV_STRING_INIT
        
    if verbose:
        print(f"\n{'='*60}")
        print(f"Target prompt: {user_prompt[:50]}...")
        print(f"Target response: {target[:50]}...")
        print(f"Initial adversarial suffix: {adv_string_init[:100]}...")
        print(f"{'='*60}")
    
    # Record start time
    start_time = time.time()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Create suffix manager
    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=adv_string_init
    )

    # Initialize parameters
    not_allowed_tokens = None if ALLOW_NON_ASCII else get_nonascii_toks(tokenizer)
    adv_suffix = adv_string_init
    suffix_length = len(tokenizer.encode(adv_suffix, add_special_tokens=False))
    mask_mlp = MaskMLP().to(device)
    
    # Use attention-guided mask initialization
    if ATTENTION_GUIDANCE_ENABLED:
        initial_mask_logits = initialize_attention_guided_mask(
            model, tokenizer, suffix_manager, adv_suffix, device, mask_mlp=mask_mlp
        )
    else:
        initial_mask_logits = torch.zeros(suffix_length, dtype=torch.float, device=device)

    mask_logits = torch.nn.Parameter(initial_mask_logits, requires_grad=True)
    current_lr = INITIAL_LR
    mask_optimizer = torch.optim.Adam([mask_logits], lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        mask_optimizer, mode='min', factor=0.8, patience=5,
        threshold=0.1, threshold_mode='abs', min_lr=1e-3
    )
    lambda_reg = LAMBDA_REG
    previous_losses = []

    # Record training process
    final_loss = float('inf')
    final_epoch = 0
    attack_success = False
    final_generation = ""
    loss_history = []
    
    try:
        for i in range(max_steps):
            if verbose and i % 10 == 0:
                print(f"Step {i+1}/{max_steps}, current suffix length: {len(tokenizer.encode(adv_suffix, add_special_tokens=False))}")
                
            temperature = 2.0 * (1 + np.cos(np.pi * i / max_steps)) / 2 + 0.1
            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
            
            # Smart pruning logic
            if SMART_PRUNING_ENABLED and i != 0 and i % PRUNING_FREQUENCY == 0:
                pruned, new_adv_suffix, new_mask_logits, keep_indices, token_to_test, pruned_mask_value, loss_increase = smart_pruning_strategy(
                    model=model,
                    input_ids=input_ids,
                    mask_logits=mask_logits,
                    control_slice=suffix_manager._control_slice,
                    target_slice=suffix_manager._target_slice,
                    loss_slice=suffix_manager._loss_slice,
                    adv_suffix=adv_suffix,
                    tokenizer=tokenizer,
                    previous_losses=previous_losses,
                    step=i,
                    min_length=MIN_SEQUENCE_LENGTH
                )

                if pruned:
                    if verbose:
                        print(f"  Pruned: {len(tokenizer.encode(adv_suffix, add_special_tokens=False))} -> {len(tokenizer.encode(new_adv_suffix, add_special_tokens=False))} tokens")
                    adv_suffix = new_adv_suffix
                    mask_logits = torch.nn.Parameter(new_mask_logits, requires_grad=True)
                    current_lr = mask_optimizer.param_groups[0]['lr']
                    mask_optimizer = torch.optim.Adam([mask_logits], lr=current_lr)

            # Update input_ids to match new adv_suffix
            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)

            # Compute coordinate gradients
            coordinate_grad, mask_grad, mask_prob = token_gradients(
                model=model,
                input_ids=input_ids,
                mask_logits=mask_logits,
                temperature=temperature,
                input_slice=suffix_manager._control_slice,
                target_slice=suffix_manager._target_slice,
                loss_slice=suffix_manager._loss_slice,
                lambda_reg=lambda_reg
            )

            # Attention guidance adjustment
            if ATTENTION_GUIDANCE_ENABLED and i % ATTENTION_GUIDANCE_FREQUENCY == 0:
                mask_logits = update_mask_with_attention_guidance(
                    model, input_ids, mask_logits,
                    suffix_manager._control_slice, suffix_manager._target_slice,
                    suffix_manager._loss_slice, guidance_strength=ATTENTION_GUIDANCE_STRENGTH
                )

                # Recompute gradients
                coordinate_grad, mask_grad, mask_prob = token_gradients(
                    model=model,
                    input_ids=input_ids,
                    mask_logits=mask_logits,
                    temperature=temperature,
                    input_slice=suffix_manager._control_slice,
                    target_slice=suffix_manager._target_slice,
                    loss_slice=suffix_manager._loss_slice,
                    lambda_reg=lambda_reg
                )

            # Update mask
            mask_optimizer.zero_grad()
            mask_logits.grad = mask_grad
            mask_optimizer.step()

            # Sample new tokens
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

            with torch.no_grad():
                # Random sample replacement batch
                new_adv_suffix_toks = sample_control(
                    adv_suffix_tokens,
                    coordinate_grad,
                    BATCH_SIZE,
                    topk=TOPK,
                    not_allowed_tokens=not_allowed_tokens
                )

                # Filter candidates
                new_adv_suffix = get_filtered_cands(
                    tokenizer,
                    new_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix
                )

                # Compute loss and select best candidate
                logits, ids = None, None
                candidate_bs = [512, 256, 128, 64]
                last_err = None
                for bs in candidate_bs:
                    try:
                        logits, ids = get_logits(
                            model=model,
                            tokenizer=tokenizer,
                            input_ids=input_ids,
                            control_slice=suffix_manager._control_slice,
                            test_controls=new_adv_suffix,
                            return_ids=True,
                            batch_size=bs
                        )
                        break
                    except RuntimeError as e:
                        last_err = e
                        if 'out of memory' in str(e).lower():
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise
                if logits is None or ids is None:
                    raise last_err if last_err is not None else RuntimeError('get_logits failed')

                losses = target_loss(logits, ids, suffix_manager._target_slice)
                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
                current_loss = losses[best_new_adv_suffix_id]
                previous_losses.append(current_loss.item())
                loss_history.append(current_loss.item())

                # Adjust learning rate
                scheduler.step(current_loss.item())
                current_lr = mask_optimizer.param_groups[0]['lr']

                # Update adversarial suffix
                adv_suffix = best_new_adv_suffix
                final_loss = current_loss.item()
                final_epoch = i + 1

                # Check attack success
                gen_str, is_success = check_for_attack_success(
                    model, tokenizer,
                    suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                    suffix_manager._assistant_role_slice,
                    TEST_PREFIXES
                )
                
                final_generation = gen_str

            if is_success:
                if verbose:
                    print(f"Attack succeeded at step {i+1}")
                attack_success = True
                break

            # Clear cache
            del coordinate_grad, adv_suffix_tokens
            gc.collect()
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error during attack: {e}")
        attack_success = False

    # Compute total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final token length
    final_token_length = len(tokenizer.encode(adv_suffix, add_special_tokens=False))
    
    # Return result
    result = {
        'user_prompt': user_prompt,
        'target': target,
        'attack_success': attack_success,
        'attack_time_seconds': total_time,
        'final_epoch': final_epoch,
        'final_token_length': final_token_length,
        'final_loss': final_loss,
        'final_suffix': adv_suffix,
        'final_generation': final_generation,
        'loss_history': loss_history
    }
    
    if verbose:
        print(f"\nAttack Results:")
        print(f"   Success: {'Yes' if attack_success else 'No'}")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Epochs: {final_epoch}")
        print(f"   Final Loss: {final_loss:.4f}")
        print(f"   Final Token Length: {final_token_length}")
        print(f"   Final Suffix: {adv_suffix}")
        print(f"   Model Response: {final_generation[:200]}...")
    
    return result


def run_batch_attack(
    model_path,
    csv_file,
    device='cuda:0',
    output_file=None,
    start_idx=0,
    end_idx=None,
    max_steps=None,
    verbose=True
):
    """
    Run batch Mask-GCG attacks
    """
    print(f"Starting batch Mask-GCG attack")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"CSV file: {csv_file}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device
    )
    conv_template = load_conversation_template(TEMPLATE_NAME)
    print("Model loaded successfully!")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} test cases")
    
    if end_idx is None:
        end_idx = len(df)
    end_idx = min(end_idx, len(df))
    
    if max_steps is None:
        max_steps = NUM_STEPS
    
    # Check if adv column exists
    has_adv_column = 'adv' in df.columns
    
    results = []
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        user_prompt = row['goal']
        target = row['target']
        adv_string_init = row['adv'] if has_adv_column and pd.notna(row['adv']) else None
        
        print(f"\n{'='*60}")
        print(f"Case {idx+1}/{end_idx}: {user_prompt[:50]}...")
        print(f"{'='*60}")
        
        result = run_single_attack(
            model=model,
            tokenizer=tokenizer,
            conv_template=conv_template,
            user_prompt=user_prompt,
            target=target,
            device=device,
            adv_string_init=adv_string_init,
            max_steps=max_steps,
            verbose=verbose
        )
        result['test_case_id'] = idx
        results.append(result)
        
        # Save results in real-time
        if output_file:
            save_results(results, output_file)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Statistics
    success_count = sum(1 for r in results if r['attack_success'])
    print(f"\nBatch test completed!")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    return results


def save_results(results, output_file):
    """Save results"""
    # JSON format
    json_file = output_file if output_file.endswith('.json') else output_file + '.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        # Remove loss_history to reduce file size
        results_to_save = [{k: v for k, v in r.items() if k != 'loss_history'} for r in results]
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    # CSV format
    csv_file = output_file.replace('.json', '.csv') if output_file.endswith('.json') else output_file + '.csv'
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'loss_history'} for r in results])
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"Results saved to: {json_file} and {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Mask-GCG: Learnable Mask GCG Adversarial Attack')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single', help='Run mode')
    parser.add_argument('--csv_file', type=str, default='data/harmful_behaviors.csv', help='Batch test CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output file name')
    parser.add_argument('--prompt', type=str, default=None, help='Target prompt for single attack')
    parser.add_argument('--target', type=str, default=None, help='Target response for single attack')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum optimization steps')
    parser.add_argument('--start_idx', type=int, default=0, help='Batch test start index')
    parser.add_argument('--end_idx', type=int, default=None, help='Batch test end index')
    
    args = parser.parse_args()
    
    # Use default values from config
    model_path = args.model_path or MODEL_PATH
    
    if args.mode == 'single':
        # Single attack mode
        user_prompt = args.prompt or USER_PROMPT
        target = args.target or TARGET
        
        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            low_cpu_mem_usage=True,
            use_cache=False,
            device=args.device
        )
        conv_template = load_conversation_template(TEMPLATE_NAME)
        print("Model loaded successfully!")
        
        result = run_single_attack(
            model=model,
            tokenizer=tokenizer,
            conv_template=conv_template,
            user_prompt=user_prompt,
            target=target,
            device=args.device,
            max_steps=args.max_steps,
            verbose=True
        )
        
        if args.output:
            save_results([result], args.output)
            
    else:
        # Batch attack mode
        output_file = args.output or f"mask_gcg_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = run_batch_attack(
            model_path=model_path,
            csv_file=args.csv_file,
            device=args.device,
            output_file=output_file,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_steps=args.max_steps,
            verbose=True
        )


if __name__ == "__main__":
    main()
