import torch
import torch.nn as nn
from fastchat.conversation import Conversation
from numpy.ma.core import indices
from torch.fx.proxy import orig_method_name

from llm_attacks import get_embedding_matrix, get_embeddings, get_embedding_layer
import numpy as np
import fastchat


def token_gradients(model, input_ids, mask_logits, temperature, input_slice, target_slice, loss_slice, lambda_reg):
    embed_weights = get_embedding_matrix(model)
    
    # Safety check: ensure input_slice length matches mask_logits length
    input_slice_length = input_ids[input_slice].shape[0]
    mask_length = len(mask_logits)
    
    if input_slice_length != mask_length:
        print(f"Warning: token_gradients dimension mismatch - input_slice_length={input_slice_length}, mask_length={mask_length}")
        print(f"   Adjusting mask_logits to match input_slice length...")
        
        # Create temporary copy to avoid modifying original Parameter
        temp_mask_logits = mask_logits.clone()
        
        if mask_length > input_slice_length:
            # mask too long, truncate
            temp_mask_logits = temp_mask_logits[:input_slice_length]
        else:
            # mask too short, pad with mean value
            mean_value = temp_mask_logits.mean()
            padding = torch.full((input_slice_length - mask_length,), 
                               mean_value, device=temp_mask_logits.device, dtype=temp_mask_logits.dtype)
            temp_mask_logits = torch.cat([temp_mask_logits, padding])
        
        # Use adjusted mask_logits
        mask_logits = temp_mask_logits
    
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()

    mask = torch.sigmoid(mask_logits / temperature)
    input_embeds = (one_hot @ embed_weights) * mask.unsqueeze(-1)
    input_embeds = input_embeds.unsqueeze(0).to(embed_weights.dtype)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    dtype = embeds.dtype
    embeds = embeds.to(dtype)
    input_embeds = input_embeds.to(dtype)

    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)
    full_embeds = full_embeds.to(dtype)
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss_token = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    reg_loss = lambda_reg * torch.mean(mask)
    # effective_tokens = torch.sum(mask > 0.1)
    # effectiveness_loss = 0.1 * torch.relu(10 - effective_tokens)

    # loss = loss_token + reg_loss + effectiveness_loss
    loss = loss_token + reg_loss
    mask_logits.grad = None
    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    mask_grad = mask_logits.grad.clone()
    return grad, mask_grad, mask


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    top_pos = top_indices[new_token_pos]
    rand_index = torch.randint(0, topk, (batch_size, 1),
                               device=grad.device)
    new_token_val = torch.gather(
        top_pos, 1,
        rand_index
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


# def sample_control(control_toks, grad, mask_logits, batch_size, topk=256, temp=1, not_allowed_tokens=None):
#     if not_allowed_tokens is not None:
#         grad[:, not_allowed_tokens.to(grad.device)] = np.infty
#
#     top_indices = (-grad).topk(topk, dim=1).indices
#     control_toks = control_toks.to(grad.device)
#
#     # Calculate positions to replace
#     mask_prob = torch.sigmoid(mask_logits)
#     replace_positions = torch.where(mask_prob < 1)[0]
#
#     # If no positions need replacement, return original tokens
#     if len(replace_positions) == 0:
#         return control_toks.repeat(batch_size, 1)
#
#     # Create batch_size copies
#     new_control_toks = control_toks.repeat(batch_size, 1)
#
#     # Select different positions for each batch sample
#     if len(replace_positions) >= batch_size:
#         # If enough active positions, select different positions for each batch
#         selected_positions = replace_positions[:batch_size]
#     else:
#         # Otherwise cycle through active positions
#         selected_positions = replace_positions.repeat(batch_size // len(replace_positions) + 1)[:batch_size]
#
#     # Randomly select new token at selected position for each batch sample
#     for i in range(batch_size):
#         pos = selected_positions[i]
#         new_token = top_indices[pos][torch.randint(0, topk, (1,), device=grad.device)]
#         new_control_toks[i, pos] = new_token
#
#     return new_control_toks

def change_lambda(lambda_reg, previous_losses,
                  window_size=5, min_lambda=0.1, max_lambda=0.6,
                  lambda_decrease_rate=0.8, lambda_increase_rate=1.2):
    # Dynamically adjust lambda_reg
    if len(previous_losses) >= window_size:
        # Keep only the most recent window_size loss values
        if len(previous_losses) > window_size:
            previous_losses.pop(0)
        # Calculate loss decrease rate
        loss_change_rate = (previous_losses[0] - previous_losses[-1]) / window_size

        # If loss decreases slowly (based on threshold)
        if loss_change_rate <= 0.012 and previous_losses[-1] > 0.5:
            # If decreasing slowly, reduce penalty
            lambda_reg = max(min_lambda, lambda_reg * lambda_decrease_rate)
        elif loss_change_rate <= 0.012 and previous_losses[-1] < 0.5:
            lambda_reg = min(max_lambda, lambda_reg * lambda_increase_rate)

    return lambda_reg, previous_losses


def change_lambda_and_lr(lambda_reg, previous_losses, current_lr, optimizer,
                         window_size=5, min_lambda=0.1, max_lambda=0.5,
                         lambda_decrease_rate=0.8, lambda_increase_rate=1.2,
                         min_lr=1e-3, max_lr=0.15, lr_increase_rate=1.2, lr_decrease_rate=0.8):
    new_lr = current_lr

    if len(previous_losses) >= window_size:
        # Keep only the most recent window_size loss values
        if len(previous_losses) > window_size:
            previous_losses.pop(0)

        # Calculate loss decrease rate
        loss_change_rate = (previous_losses[0] - previous_losses[-1]) / window_size

        # If loss decreases slowly (based on threshold)
        if loss_change_rate <= 0.02:
            # Slow decrease: reduce penalty, increase learning rate
            lambda_reg = max(min_lambda, lambda_reg * lambda_decrease_rate)
            new_lr = min(max_lr, current_lr * lr_increase_rate)

            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        elif loss_change_rate > 0.05:
            # Fast decrease: increase penalty, reduce learning rate to prevent overfitting
            lambda_reg = min(max_lambda, lambda_reg * lambda_increase_rate)
            new_lr = max(min_lr, current_lr * lr_decrease_rate)

            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    return lambda_reg, previous_losses, new_lr


def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if not hasattr(conv_template, 'system'):
        conv_template.system = ""
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        elif self.conv_template.name in ['llama-3', 'llama-3.1', 'llama3', 'llama3.1']:
            # Llama 3.1 uses special tokens: <|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>
            self.conv_template.messages = []

            # Build the prompt step by step to track token positions
            # Start with begin_of_text token (if present in template)
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            # Add instruction only
            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            # Add instruction + adversarial string
            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            # Add assistant role header
            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            # Add target response
            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

            # Adjust for special tokens at the end
            # Llama 3.1 typically ends with <|eot_id|> token
            special_token_offset = 1 if self.tokenizer.eos_token_id in toks[-2:] else 0
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - special_token_offset)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - special_token_offset - 1)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))

                separator = ' ' if self.instruction else ''
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

    def get_suffix_ids(self, suffixs):
        batch_toks = [
            torch.tensor(self.tokenizer(s, add_special_tokens=False).input_ids[:self._target_slice.stop])
            for s in suffixs
        ]
        input_toks = torch.stack(batch_toks)
        return input_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=False, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=False)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(
                    control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        if not cands:
            cands = [curr_control] * len(control_cand)
        else:
            cands += [cands[-1]] * (len(control_cand) - len(cands))
    return cands



def normalize_attention_weights(attention_weights, device):
    """Stable normalization of attention weights (optimized version)"""
    value_range = attention_weights.max() - attention_weights.min()
    mean_value = attention_weights.mean()
    std_value = attention_weights.std()

    # Debug output removed for performance
    # print(f"   Attention stats: range={value_range:.6f}, mean={mean_value:.6f}, std={std_value:.6f}")

    if value_range > 1e-6:
        # Meaningful numerical variation, use improved normalization
        # First perform min-max normalization to [0,1]
        attention_weights = (attention_weights - attention_weights.min()) / (value_range+ 1e-8)
        # Prevent over-extremization: mix with uniform distribution
        uniform_weights = torch.ones_like(attention_weights) / len(attention_weights)
        attention_weights = 0.9 * attention_weights + 0.1 * uniform_weights

    elif value_range > 1e-8:
        # Small variation, use mild softmax
        # Z-score normalization followed by softmax, temperature controls smoothness
        z_scores = (attention_weights - mean_value) / (std_value + 1e-8)
        temperature = 1.0  # Adjustable temperature
        attention_weights = torch.softmax(z_scores / temperature, dim=0)

    else:
        # Almost no variation, use slightly center-biased uniform distribution
        control_length = len(attention_weights)

        # Create slightly center-biased distribution
        positions = torch.arange(control_length, device=device, dtype=torch.float)
        center = control_length / 2.0
        # Use smaller decay coefficient for flatter distribution
        weights = torch.exp(-0.05 * (positions - center) ** 2)

        # Add small uniform noise to avoid over-determinism
        noise = torch.rand_like(weights) * 0.1
        weights = weights + noise

        attention_weights = weights / weights.sum()

    # Debug output removed
    # print(f"   Final attention weights range: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]")
    # print(f"   Final attention weights distribution - mean: {attention_weights.mean():.6f}, std: {attention_weights.std():.6f}")
    return attention_weights


def get_attention_scores(model, input_ids, control_slice, target_slice):
    """
    Get attention scores of suffix tokens to target tokens
    """
    # Save original mode and set to evaluation mode
    original_training_mode = model.training
    model.eval()

    control_length = control_slice.stop - control_slice.start

    try:
        with torch.no_grad():
            # Forward pass to get attention weights
            outputs = model(input_ids.unsqueeze(0), output_attentions=True)
            attentions = outputs.attentions  # tuple of attention matrices for each layer

            # Check if attentions is None or empty
            if attentions is None or len(attentions) == 0:
                print("Warning: Model did not return attention weights. Using uniform initialization.")
                # Fallback: return uniform distribution
                attention_weights = torch.ones(control_length, device=input_ids.device) / control_length
                return attention_weights

            # Aggregate attention from all layers, focusing on last few layers
            num_layers = len(attentions)
            attention_weights = torch.zeros(control_length, device=input_ids.device)

            # Use weighted average of last 3 layers, with increasing weights
            layer_weights = [0.2, 0.3, 0.5]  # Later layers have higher weights
            selected_layers = list(range(max(0, num_layers - 3), num_layers))

            for i, layer_idx in enumerate(selected_layers):
                # attention shape: [batch_size, num_heads, seq_len, seq_len]
                layer_attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]

                # Average all attention heads
                avg_attention = layer_attention.mean(dim=0)  # [seq_len, seq_len]

                # Strategy 2: Attention from target tokens to control tokens (reverse importance)
                target_to_control = avg_attention[target_slice.start:target_slice.stop,
                                    control_slice.start:control_slice.stop]
                if target_to_control.numel() > 0:
                    strategy2_scores = target_to_control.mean(dim=0)  # Average over target positions
                else:
                    strategy2_scores = torch.zeros(control_length, device=input_ids.device)

                # Strategy 3: Global attention strength of control tokens (excluding self-attention)
                control_attention = avg_attention[control_slice.start:control_slice.stop, :]
                # Create mask to exclude self-attention
                global_mask = torch.ones_like(control_attention)
                global_mask[:, control_slice.start:control_slice.stop] = 0  # Exclude self-attention
                masked_attention = control_attention * global_mask
                strategy3_scores = masked_attention.sum(dim=1) / (global_mask.sum(dim=1) + 1e-8)

                # Combined strategy: dynamic weight allocation
                total_signal = strategy2_scores.sum() + strategy3_scores.sum()
                if total_signal > 1e-8:
                    # Dynamically allocate weights based on effectiveness of each strategy
                    w2 = strategy2_scores.sum() / total_signal
                    w3 = strategy3_scores.sum() / total_signal
                    combined_scores = w2 * strategy2_scores + w3 * strategy3_scores
                else:
                    combined_scores = torch.ones(control_length, device=input_ids.device) / control_length

                attention_weights += layer_weights[i] * combined_scores

            # Normalization strategy
            attention_weights = normalize_attention_weights(attention_weights, input_ids.device)

        return attention_weights

    except Exception as e:
        print(f"Warning: Failed to get attention scores: {e}. Using uniform initialization.")
        # Fallback: return uniform distribution
        attention_weights = torch.ones(control_length, device=input_ids.device) / control_length
        return attention_weights

    finally:
        # Restore original training mode
        model.train(original_training_mode)

class MaskMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, scores):  # scores: [seq_len]
        return self.mlp(scores.unsqueeze(-1)).squeeze(-1)  # Output logits


def initialize_attention_guided_mask(model, tokenizer, suffix_manager, adv_suffix, device, mask_mlp=None):
    """
    Initialize mask using attention scores (via learnable MLP transform: attention -> logit)
    """
    print("Computing attention-guided mask initialization (MLP version)...")

    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice

    attention_scores = get_attention_scores(model, input_ids, control_slice, target_slice)
    print(f"Attention score range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")

    # Note: You must pass a mask_mlp (learnable transformer), otherwise use default (non-training state)
    if mask_mlp is None:
        print("Warning: mask_mlp not provided, using default initialization (no training)")
        mask_mlp = MaskMLP().to(device)
    scaled_attention = (attention_scores - attention_scores.mean()) / (attention_scores.std() + 1e-6)
    scaled_attention = scaled_attention * 4  # Empirical value, adjustable

    with torch.no_grad():  # No training during initialization
        initial_logits = mask_mlp(scaled_attention)

    # Add perturbation to avoid fixed mask
    noise = torch.randn_like(initial_logits) * 0.1
    initial_logits += noise

    # Print statistics
    print(f"Initial logits range: [{initial_logits.min():.4f}, {initial_logits.max():.4f}]")
    print(f"Corresponding probability range: [{torch.sigmoid(initial_logits).min():.4f}, {torch.sigmoid(initial_logits).max():.4f}]")

    # Token visualization
    suffix_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(adv_suffix, add_special_tokens=False)
    )
    print("\nAttention-guided + MLP initial token weights:")
    for i, (token, att_score, final_prob) in enumerate(
        zip(suffix_tokens, attention_scores, torch.sigmoid(initial_logits))
    ):
        print(f"{i:2d}. {token:>15} | Attention:{att_score:.3f} | Mask prob:{final_prob:.3f}")

    return initial_logits



def update_mask_with_attention_guidance(model, input_ids, mask_logits, control_slice, target_slice,
                                        loss_slice, guidance_strength=0.1):
    """
    Use attention information to guide mask updates during training
    """
    with torch.no_grad():

        current_attention = get_attention_scores(model, input_ids, control_slice, target_slice)

        # Get current mask probability
        current_mask_prob = torch.sigmoid(mask_logits)

        # Calculate difference between attention and mask
        attention_mask_diff = current_attention - current_mask_prob

        # Adaptive guidance strength: larger difference = stronger guidance
        adaptive_strength = guidance_strength * (1 + torch.abs(attention_mask_diff).mean())

        print(f"   Attention guidance: base_strength={guidance_strength:.3f}, adaptive_strength={adaptive_strength:.3f}")
        print(f"   Mean difference: {torch.abs(attention_mask_diff).mean():.3f}")

        # Smart guidance: stronger adjustment for tokens with larger differences
        token_wise_guidance = adaptive_strength * attention_mask_diff

        # Add smoothing constraint to prevent drastic changes
        max_change = 0.5  # Limit maximum change per step
        token_wise_guidance = torch.clamp(token_wise_guidance, -max_change, max_change)

        # Apply guidance
        old_logits = mask_logits.data.clone()
        mask_logits.data += token_wise_guidance

        # Stability check: ensure logits don't become too extreme
        mask_logits.data = torch.clamp(mask_logits.data, -10, 10)

        actual_change = (mask_logits.data - old_logits).abs().mean()
        print(f"   Actual mask change magnitude: {actual_change:.4f}")

    return mask_logits


def visualize_attention_and_mask(tokenizer, adv_suffix, attention_scores, mask_probs, step=0):
    """
    Visualize attention scores and mask probabilities
    """
    suffix_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(adv_suffix, add_special_tokens=False)
    )

    print(f"\nStep {step} - Attention and mask visualization:")
    print("=" * 80)
    print(f"{'Token':>15} | {'Attention':>8} | {'MaskProb':>8} | {'Diff':>8} | {'Visual':>20}")
    print("-" * 80)

    for i, (token, att_score, mask_prob) in enumerate(
            zip(suffix_tokens, attention_scores, mask_probs)
    ):
        diff = abs(att_score - mask_prob)

        # Create simple visualization bar chart
        att_bar = '#' * int(att_score * 10) + '.' * (10 - int(att_score * 10))
        mask_bar = '*' * int(mask_prob * 10) + '.' * (10 - int(mask_prob * 10))

        print(f"{token:>15} | {att_score:8.3f} | {mask_prob:8.3f} | {diff:8.3f} | A:{att_bar[:8]} M:{mask_bar[:8]}")

    print("=" * 80)
    print("A=Attention, M=Mask, #=high, *=medium, .=low")


def analyze_token_importance_trends(attention_history, mask_history, tokens):
    """
    Analyze token importance change trends
    """
    if len(attention_history) < 2:
        return

    print(f"\nToken importance trend analysis (last {len(attention_history)} steps):")
    print("=" * 60)

    for i, token in enumerate(tokens):
        att_trend = attention_history[-1][i] - attention_history[0][i]
        mask_trend = mask_history[-1][i] - mask_history[0][i]

        att_arrow = "UP" if att_trend > 0.05 else "DOWN" if att_trend < -0.05 else "STABLE"
        mask_arrow = "UP" if mask_trend > 0.05 else "DOWN" if mask_trend < -0.05 else "STABLE"

        print(f"{token:>15} | Attention {att_arrow} {att_trend:+.3f} | Mask {mask_arrow} {mask_trend:+.3f}")


def smart_pruning_strategy(model, input_ids, mask_logits, control_slice, target_slice,
                           loss_slice, adv_suffix, tokenizer, previous_losses,
                           step, min_length=10,threshold=0.2):

    current_length = len(mask_logits)

    # 1. Length check: too short to prune
    if current_length <= min_length:
        return False, adv_suffix, mask_logits, None, None, None, None

    # 2. Simplified pruning timing: check based on config frequency except step 0
    from config import PRUNING_FREQUENCY
    if step == 0 or step % PRUNING_FREQUENCY != 0:
        return False, adv_suffix, mask_logits, None, None, None, None

    print(f"Step {step}: Executing smart pruning check")

    # 3. Use mask probability directly as importance score
    mask_probs = torch.sigmoid(mask_logits)
    importance_scores = mask_probs
    
    print(f"Mask probability range: [{importance_scores.min():.4f}, {importance_scores.max():.4f}]")


    # Find low importance tokens (low mask values)
    low_importance_indices = torch.where(importance_scores < threshold)[0]
    if len(low_importance_indices) == 0:
        print("No low importance tokens found, skipping pruning")
        return False, adv_suffix, mask_logits, None, None, None, None

    # 5. Select the token with lowest importance for testing
    lowest_importance_idx = low_importance_indices[torch.argmin(importance_scores[low_importance_indices])]
    token_to_test = lowest_importance_idx.item()
    pruned_mask_value = importance_scores[token_to_test].item()
    
    print(f"Testing removal of token {token_to_test}, mask value: {importance_scores[token_to_test]:.4f}")

    # 6. Loss verification: test if removing this token increases loss too much
    try:
        # Calculate current loss as baseline
        from llm_attacks.minimal_gcg.opt_utils import target_loss, get_logits
        
        # Current loss
        with torch.no_grad():
            current_logits = model(input_ids.unsqueeze(0)).logits
            current_loss = target_loss(
                current_logits, 
                input_ids.unsqueeze(0), 
                target_slice
            ).item()
        
        # Build new sequence after removing the token
        suffix_token_ids = tokenizer.encode(adv_suffix, add_special_tokens=False)
        keep_positions = [i for i in range(len(suffix_token_ids)) if i != token_to_test]

        if len(keep_positions) < min_length:
             print(f"Warning: Length too short after removal, skipping")
             return False, adv_suffix, mask_logits, None, None, None, None
        
        # Build new suffix
        new_suffix_ids = [suffix_token_ids[i] for i in keep_positions]
        new_adv_suffix = tokenizer.decode(new_suffix_ids, clean_up_tokenization_spaces=False)
        # Verify new suffix encoding length matches retained token count
        re_encoded_ids = tokenizer.encode(new_adv_suffix, add_special_tokens=False)
        if len(re_encoded_ids) != len(keep_positions):
            print(f"Warning: Token length mismatch: expected {len(keep_positions)}, got {len(re_encoded_ids)}")
            # Use ID list directly to avoid encode/decode issues
            new_adv_suffix = tokenizer.decode(new_suffix_ids, skip_special_tokens=True)
            # If still not working, abort pruning
            re_encoded_ids = tokenizer.encode(new_adv_suffix, add_special_tokens=False)
            if len(re_encoded_ids) != len(keep_positions):
                print(f"Error: Cannot maintain token length consistency, aborting pruning")
                return False, adv_suffix, mask_logits, None, None, None, None
        # Build new input_ids for testing
        # Need to rebuild complete sequence from suffix_manager
        # SuffixManager is defined in this file
        # Create temporary suffix_manager for testing
        temp_input_ids = input_ids.clone()
        
        # Directly replace control portion
        new_suffix_tensor = torch.tensor(
            tokenizer.encode(new_adv_suffix, add_special_tokens=False), 
            device=input_ids.device
        )
        
        # Build new complete input
        new_input_ids = torch.cat([
            temp_input_ids[:control_slice.start],
            new_suffix_tensor,
            temp_input_ids[control_slice.stop:]
        ])
        
        # Adjust target_slice and loss_slice to match new length
        length_diff = len(new_suffix_tensor) - (control_slice.stop - control_slice.start)
        new_target_slice = slice(target_slice.start + length_diff, target_slice.stop + length_diff)
        
        # Calculate loss after removal
        with torch.no_grad():
            new_logits = model(new_input_ids.unsqueeze(0)).logits
            new_loss = target_loss(
                new_logits,
                new_input_ids.unsqueeze(0), 
                new_target_slice
            ).item()
        
        # Check if loss change is acceptable
        loss_increase = new_loss - current_loss
        max_allowed_increase = 0.1  # Allow maximum loss increase of 0.1
        
        print(f"Loss change: {current_loss:.4f} -> {new_loss:.4f} (increase: {loss_increase:.4f})")

        if loss_increase > max_allowed_increase:
             print(f"Error: Loss increase too high ({loss_increase:.4f} > {max_allowed_increase}), aborting pruning")
             return False, adv_suffix, mask_logits, None, None, pruned_mask_value, loss_increase
        
         # 7. Execute actual pruning
        keep_indices = torch.tensor(keep_positions, device=mask_logits.device)
        new_mask_logits = mask_logits[keep_indices].clone()
         

        removed_token = tokenizer.convert_ids_to_tokens([suffix_token_ids[token_to_test]])[0]
        print(f"Successfully pruned token: {removed_token}")
        print(f"   Mask value: {importance_scores[token_to_test]:.4f}")
        print(f"   Length change: {current_length} -> {len(new_mask_logits)}")
        print(f"   Loss change: +{loss_increase:.4f}")
         
        return True, new_adv_suffix, new_mask_logits, keep_indices, token_to_test, pruned_mask_value, loss_increase
        
    except Exception as e:
        print(f"Warning: Pruning validation failed: {e}")
        return False, adv_suffix, mask_logits, None, None, None, None


def preserve_optimizer_state(old_optimizer, new_mask_logits, keep_indices, old_scheduler=None):
    """
    Simple and effective optimizer state preservation method
    Saves Adam's key states: exp_avg, exp_avg_sq and step
    """
    print("Preserving optimizer state...")
    
    # Get current learning rate and other parameters
    current_lr = old_optimizer.param_groups[0]['lr']
    
    # Create new optimizer
    new_optimizer = torch.optim.Adam([new_mask_logits], lr=current_lr)
    
    try:
        # Get old optimizer's state
        old_state_dict = old_optimizer.state_dict()
        
        # Check if there are states to save
        if len(old_state_dict['state']) > 0:
            # Get the first (and only) parameter's state
            old_param_id = list(old_state_dict['state'].keys())[0]
            old_state = old_state_dict['state'][old_param_id]
            
            # Check if Adam states exist
            if 'exp_avg' in old_state and 'exp_avg_sq' in old_state:
                # Extract retained states based on keep_indices
                old_exp_avg = old_state['exp_avg'][keep_indices].clone()
                old_exp_avg_sq = old_state['exp_avg_sq'][keep_indices].clone()
                old_step = old_state.get('step', 0)
                
                print(f"   Saving state: step={old_step}, retaining {len(keep_indices)} positions")
                
                # Initialize new optimizer (trigger state creation)
                dummy_grad = torch.zeros_like(new_mask_logits)
                new_mask_logits.grad = dummy_grad
                new_optimizer.step()
                new_optimizer.zero_grad()
                new_mask_logits.grad = None
                
                # Update new optimizer's state
                new_state_dict = new_optimizer.state_dict()
                new_param_id = list(new_state_dict['state'].keys())[0]
                
                # Replace with saved states
                new_state_dict['state'][new_param_id]['exp_avg'] = old_exp_avg
                new_state_dict['state'][new_param_id]['exp_avg_sq'] = old_exp_avg_sq
                new_state_dict['state'][new_param_id]['step'] = old_step
                
                # Load updated state
                new_optimizer.load_state_dict(new_state_dict)
                
                print(f"   Optimizer state saved successfully")
            else:
                print("   Warning: No Adam state to save (might be first step)")
        else:
            print("   Warning: Optimizer state is empty")
    
    except Exception as e:
        print(f"   Warning: State preservation failed, using default initialization: {e}")
    
    # Rebuild learning rate scheduler
    new_scheduler = None
    if old_scheduler is not None:
        try:
            from torch import optim
            new_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                new_optimizer, mode='min', factor=0.8, patience=5,
                threshold=0.1, threshold_mode='abs', min_lr=1e-3
            )
            # Try to preserve scheduler's internal state
            if hasattr(old_scheduler, 'num_bad_epochs'):
                new_scheduler.num_bad_epochs = old_scheduler.num_bad_epochs
            if hasattr(old_scheduler, 'best'):
                new_scheduler.best = old_scheduler.best
            print(f"   Scheduler state also saved")
        except Exception as e:
            print(f"   Warning: Scheduler state preservation failed: {e}")
    
    return new_optimizer, new_scheduler
