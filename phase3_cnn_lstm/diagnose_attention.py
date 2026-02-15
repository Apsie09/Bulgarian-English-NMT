# diagnose_attention.py - Check if model is attending to source

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pickle
import sys
from models.convenc_lstmdec import ConvEncoderLSTMDecoder
from shared.parameters import (phase3_d_model, phase3_hidden_size, phase3_enc_layers,
                       phase3_dec_layers, phase3_kernel_size, phase3_max_len,
                       phase3_dropout, device, modelSeq2SeqFileName, wordsFileName)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "shared"))

# Load vocabulary and model
word2ind = pickle.load(open(ROOT / wordsFileName, 'rb'))
ind2word = {v: k for k, v in word2ind.items()}
vocab_size = len(word2ind)

_bpe_candidates = [
    ROOT / 'bpeModel',
    ROOT / 'phase2_transformer_bpe' / 'bpeModel',
    ROOT / 'shared' / 'bpeModel',
]
bpe_path = next((p for p in _bpe_candidates if p.exists()), None)
if bpe_path is None:
    raise FileNotFoundError(
        "bpeModel not found. Tried: " + ", ".join(str(p) for p in _bpe_candidates)
    )
bpe = pickle.load(open(bpe_path, 'rb'))
old_to_new = bpe._old_to_new

model = ConvEncoderLSTMDecoder(
    vocab_size=vocab_size,
    d_model=phase3_d_model,
    hidden_size=phase3_hidden_size,
    enc_layers=phase3_enc_layers,
    dec_layers=phase3_dec_layers,
    kernel_size=phase3_kernel_size,
    max_len=phase3_max_len,
    dropout=0.0
).to(device)
model.load(str(ROOT / modelSeq2SeqFileName))
model.eval()

# Test sentences
test_sentences = [
    "През август Съветът на гуверньорите на Международния валутен фонд одобри 250 млрд. щатски долара като общо разпределение от специалните права на тираж на Международния валутен фонд, от които 18 млрд. щатски долара ще бъдат за държавите с нисък доход, а Международният валутен фонд ще бъде призован да даде отчет в Питсбърг относно други мерки за държавите с нисък доход.",
    "Ако го забраним като добавка в храни, можем ли да продължим да съществуваме или ще трябва постепенно да се самоотстраним като опасни отпадъци?",
]

print("=" * 80)
print("ATTENTION WEIGHT DIAGNOSTICS")
print("=" * 80)

for sent_idx, src_text in enumerate(test_sentences):
    print(f"\n{'='*80}")
    print(f"Sentence {sent_idx + 1}")
    print(f"{'='*80}")
    print(f"Source: {src_text[:100]}...")
    
    # Tokenize
    src_tokens = [old_to_new[t] for t in bpe.to_tokens(src_text)]
    src_ids = [0] + src_tokens + [1]  # <S> ... </S>
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.ones(1, len(src_ids), dtype=torch.bool, device=device)
    
    # Encode
    Z, V = model.encoder(src_tensor, src_mask)
    
    # Decode step by step and collect attention weights
    B = 1
    hidden = model.decoder._init_hidden(B, device)
    context = torch.zeros(B, model.d_model, device=device)
    current_token = torch.tensor([0], dtype=torch.long, device=device)  # <S>
    
    attention_weights_list = []
    output_tokens = []
    
    max_decode_steps = 20  # Only decode first 20 tokens
    
    for step in range(max_decode_steps):
        g_t = model.decoder.embedding(current_token)
        h_prev = hidden[0][-1]
        
        # Get attention weights
        context, attn_weights = model.decoder.attention(h_prev, g_t, Z, V, src_mask)
        # Line 75 - add .detach()
        attention_weights_list.append(attn_weights[0].detach().cpu().numpy())        
        # Continue decoding
        lstm_input = torch.cat([g_t, context], dim=-1).unsqueeze(1)
        lstm_out, hidden = model.decoder.lstm(lstm_input, hidden)
        h_t = lstm_out.squeeze(1)
        
        combined = torch.cat([h_t, context], dim=-1)
        o_t = torch.tanh(model.decoder.W_cat(combined))
        logits_t = model.decoder.W_out(o_t)
        
        next_token = logits_t.argmax(dim=-1)
        token_id = next_token.item()
        
        if token_id == 1:  # </S>
            break
            
        output_tokens.append(token_id)
        current_token = next_token
    
    # Show output
    output_text = ''.join(ind2word.get(t, '?') for t in output_tokens)
    print(f"\nOutput: {output_text[:100]}...")
    
    # Analyze attention weights
    import numpy as np
    attn_array = np.array(attention_weights_list)  # (decode_steps, src_len)
    
    print(f"\n{'Attention Statistics':^80}")
    print(f"{'-'*80}")
    print(f"Attention shape: {attn_array.shape} (decode_steps × src_len)")
    print(f"Mean attention weight: {attn_array.mean():.6f}")
    print(f"Std attention weight: {attn_array.std():.6f}")
    print(f"Max attention weight: {attn_array.max():.6f}")
    print(f"Min attention weight: {attn_array.min():.6f}")
    
    # Check if attention is uniform (indicates not learning)
    uniform_value = 1.0 / len(src_ids)
    mean_deviation = np.abs(attn_array - uniform_value).mean()
    print(f"\nExpected uniform weight: {uniform_value:.6f}")
    print(f"Mean deviation from uniform: {mean_deviation:.6f}")
    
    if mean_deviation < 0.01:
        print("WARNING: Attention is nearly UNIFORM - model is NOT attending!")
    else:
        print("✓ Attention has some variation")
    
    # Check entropy (high entropy = flat distribution)
    entropies = -np.sum(attn_array * np.log(attn_array + 1e-10), axis=1)
    max_entropy = np.log(len(src_ids))
    print(f"\nMean attention entropy: {entropies.mean():.3f}")
    print(f"Max possible entropy: {max_entropy:.3f} (completely uniform)")
    print(f"Entropy ratio: {entropies.mean()/max_entropy:.2%}")
    
    if entropies.mean() / max_entropy > 0.9:
        print("WARNING: Very high entropy - attention is almost FLAT!")
    
    # Show first few decode steps
    print(f"\n{'First 5 Decode Steps':^80}")
    print(f"{'-'*80}")
    src_words = [ind2word.get(t, '?') for t in src_ids]
    
    for step in range(min(5, len(attention_weights_list))):
        weights = attention_weights_list[step]
        top_5_idx = np.argsort(weights)[-5:][::-1]
        
        out_word = ind2word.get(output_tokens[step] if step < len(output_tokens) else 0, '?')
        print(f"\nStep {step+1}: Generated '{out_word}'")
        print(f"  Top 5 attended source positions:")
        for i, idx in enumerate(top_5_idx):
            print(f"    {i+1}. pos {idx:3d} (weight={weights[idx]:.4f}): '{src_words[idx]}'")

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*80}\n")
