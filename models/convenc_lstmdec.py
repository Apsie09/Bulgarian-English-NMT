import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# CNN stack for providing residual connections in the encoder

class ConvEncoderStack(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        padding = (kernel_size - 1) // 2

        for _ in range(num_layers):
            self.layers.append(
                nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)

        for conv in self.layers:
            residual = x
            x = conv(x)
            x = torch.tanh(x + residual)
            x = self.dropout(x)
        
        return x.transpose(1, 2)  # (batch_size, seq_len, d_model)

# CNN Encoder

class ConvEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, 
                 kernel_size: int = 3, num_layers: int = 4, 
                 dropout: float = 0.1, pad_idx: int = 3):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.cnn_z = ConvEncoderStack(d_model, kernel_size, num_layers, dropout) # For Attention Scoring
        self.cnn_v = ConvEncoderStack(d_model, kernel_size, num_layers, dropout) # For Attention Values

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.word_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.1)

        with torch.no_grad():
            self.word_embedding.weight[self.pad_idx].fill_(0)
    
    def forward(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = src_ids.shape
        device = src_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        word_emb = self.word_embedding(src_ids)
        pos_emb = self.pos_embedding(positions)
        e = self.dropout(word_emb + pos_emb)

        Z = self.cnn_z(e) # For Attention Scoring
        V = self.cnn_v(e) # For Attention Values

        mask = src_pad_mask.unsqueeze(-1).float()  # (B, S, 1)
        Z = Z * mask
        V = V * mask

        return Z, V

# Dot-Product Attention

class LuongAttention(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.W_d = nn.Linear(hidden_size, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_dropout_p = dropout
    
    def forward(self, decoder_hidden: torch.Tensor, target_emb: torch.Tensor,
                Z: torch.Tensor, V: torch.Tensor, src_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        d_i = self.W_d(decoder_hidden) + target_emb  # (B, d_model)

        scores = torch.bmm(d_i.unsqueeze(1), Z.transpose(1, 2)).squeeze(1)  # (B, S)
        scores = scores / (self.d_model ** 0.5)  # SCALED DOT-PRODUCT

        scores = scores.masked_fill(~src_pad_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, S)
        if self.attn_dropout_p > 0:
            attn_weights = self.attn_dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        context = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # (B, d_model)

        return context, attn_weights

# LSTM Decoder with Luong Attention
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, 
                 num_layers: int = 2, dropout: float = 0.1, pad_idx: int = 3):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        self.attention = LuongAttention(d_model, hidden_size, dropout)

        # LSTM - input is [target_emb; context] = d_model + d_model

        self.lstm = nn.LSTM(
            input_size=d_model + d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.W_cat = nn.Linear(hidden_size + d_model, d_model)
        self.W_out = nn.Linear(d_model, vocab_size)

        self.h_init = nn.Linear(d_model, hidden_size)
        self.c_init = nn.Linear(d_model, hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].fill_(0)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, target_in_ids: torch.Tensor, Z: torch.Tensor, V: torch.Tensor,
                src_pad_mask: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                enc_summary: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        B, T = target_in_ids.shape
        device = target_in_ids.device

        g = self.dropout(self.embedding(target_in_ids))  # (B, T, d_model)

        if hidden is None:
            if enc_summary is not None:
                hidden = self._init_hidden_from_enc(enc_summary)
            else:
                hidden = self._init_hidden(B, device)
        
        all_logits = []

        h_prev = hidden[0][-1]  

        for t in range(T):
            g_t = g[:, t, :]

            context, _ = self.attention(h_prev, g_t, Z, V, src_pad_mask)

            lstm_input = torch.cat([g_t, context], dim=-1).unsqueeze(1)  # (B, 1, d_model + d_model)

            lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: (B, 1, hidden_size)
            h_t = lstm_out.squeeze(1)  # (B, hidden_size)

            h_prev = h_t

            combined = torch.cat([h_t, context], dim=-1)  # (B, hidden_size + d_model)
            o_t = torch.tanh(self.W_cat(combined))  # (B, d_model)
            logits_t = self.W_out(o_t)  # (B, vocab_size)

            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)  # (B, T, vocab_size)
        return logits, hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)

    def _init_hidden_from_enc(self, enc_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_summary: (B, d_model)
        h = torch.tanh(self.h_init(enc_summary))
        c = torch.tanh(self.c_init(enc_summary))
        h_0 = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_0 = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return (h_0, c_0)
    
    def generate_step(self, token_id: torch.Tensor, context: torch.Tensor,
                    Z: torch.Tensor, V: torch.Tensor, src_pad_mask: torch.Tensor,
                    hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        
        g_t = self.embedding(token_id)  # (B, d_model)
        
        h_prev = hidden[0][-1]  # (B, hidden_size)
        
        context, _ = self.attention(h_prev, g_t, Z, V, src_pad_mask)
        
        lstm_input = torch.cat([g_t, context], dim=-1).unsqueeze(1)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        h_t = lstm_out.squeeze(1)
        
        combined = torch.cat([h_t, context], dim=-1)
        o_t = torch.tanh(self.W_cat(combined))
        logits_t = self.W_out(o_t)
        
        return logits_t, context, hidden    

# Complete ConvEncoder + LSTMDecoder Model

class ConvEncoderLSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, hidden_size: int = 512, 
                 enc_layers: int = 4, dec_layers: int = 2, kernel_size: int = 3,
                 max_len: int = 512, dropout: float = 0.1, use_enc_init: bool = True):
        super().__init__()

        self.start_idx = 0
        self.end_idx = 1
        self.unk_idx = 2
        self.pad_idx = 3
        self.trans_idx = 4

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_enc_init = use_enc_init

        self.encoder = ConvEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            kernel_size=kernel_size,
            num_layers=enc_layers,
            dropout=dropout,
            pad_idx=self.pad_idx
        )

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_size=hidden_size,
            num_layers=dec_layers,
            dropout=dropout,
            pad_idx=self.pad_idx
        )
    
    def forward(self, src_ids: torch.Tensor, target_in_ids: torch.Tensor,
                 src_pad_mask: torch.Tensor) -> torch.Tensor:
        
        Z, V = self.encoder(src_ids, src_pad_mask)

        enc_summary = self._encoder_summary(V, src_pad_mask) if self.use_enc_init else None
        logits, _ = self.decoder(target_in_ids, Z, V, src_pad_mask, enc_summary=enc_summary)

        return logits

    def compute_loss(self, src_ids: torch.Tensor, target_in_ids: torch.Tensor,
                     target_out_ids: torch.Tensor, src_pad_mask: torch.Tensor, 
                     target_pad_mask: torch.Tensor) -> torch.Tensor:
        
        logits = self.forward(src_ids, target_in_ids, src_pad_mask)

        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = target_out_ids.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_idx, label_smoothing=0.0)

        return loss

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor,
                    max_len: int = 100, repetition_penalty: float = 1.2,
                    block_ngram_repeat: int = 3) -> List[List[int]]:

        self.eval()
        B = src_ids.size(0)
        device = src_ids.device
        
        Z, V = self.encoder(src_ids, src_pad_mask)
        enc_summary = self._encoder_summary(V, src_pad_mask) if self.use_enc_init else None
        
        hidden = self.decoder._init_hidden_from_enc(enc_summary) if enc_summary is not None else self.decoder._init_hidden(B, device)
        context = torch.zeros(B, self.d_model, device=device)
        current_token = torch.full((B,), self.start_idx, dtype=torch.long, device=device)
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results = [[] for _ in range(B)]
        token_history = [set() for _ in range(B)]
        
        for _ in range(max_len):
            logits, context, hidden = self.decoder.generate_step(
                current_token, context, Z, V, src_pad_mask, hidden
            )
            
            if repetition_penalty != 1.0:
                for b in range(B):
                    for token_id in token_history[b]:
                        logits[b, token_id] = logits[b, token_id] / repetition_penalty
            
            if block_ngram_repeat > 0:
                for b in range(B):
                    if len(results[b]) >= block_ngram_repeat - 1:
                        prefix = tuple(results[b][-(block_ngram_repeat-1):])
                        banned_tokens = set()
                        for i in range(len(results[b]) - block_ngram_repeat + 1):
                            ngram = tuple(results[b][i:i+block_ngram_repeat])
                            if ngram[:block_ngram_repeat-1] == prefix:
                                banned_tokens.add(ngram[-1])
                        for token_id in banned_tokens:
                            logits[b, token_id] = float('-inf')
            
            next_token = logits.argmax(dim=-1)  
            
            for b in range(B):
                if not finished[b]:
                    token = next_token[b].item()
                    if token == self.end_idx:
                        finished[b] = True
                    else:
                        results[b].append(token)
                        token_history[b].add(token)
                
            if finished.all():
                break
            
            current_token = next_token
        
        return results

    @torch.no_grad()
    def beam_search(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor,
                    beam_size: int = 4, max_len: int = 100) -> List[List[int]]:
       
        self.eval()
        B = src_ids.size(0)
        device = src_ids.device
        
        all_results = []
        
        for b in range(B):
            src_b = src_ids[b:b+1]  # [1, S]
            mask_b = src_pad_mask[b:b+1]  # [1, S]
            
            Z, V = self.encoder(src_b, mask_b)
            enc_summary = self._encoder_summary(V, mask_b) if self.use_enc_init else None
            
            Z = Z.expand(beam_size, -1, -1)  # [beam, S, d]
            V = V.expand(beam_size, -1, -1)
            mask_b = mask_b.expand(beam_size, -1)
            
            if enc_summary is not None:
                enc_summary = enc_summary.expand(beam_size, -1)
                hidden = self.decoder._init_hidden_from_enc(enc_summary)
            else:
                hidden = self.decoder._init_hidden(beam_size, device)
            context = torch.zeros(beam_size, self.d_model, device=device)
            current_tokens = torch.full((beam_size,), self.start_idx, dtype=torch.long, device=device)
            
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = float('-inf')  
            
            completed = []
            
            sequences = [[] for _ in range(beam_size)]
            
            for step in range(max_len):
                logits, context, hidden = self.decoder.generate_step(
                    current_tokens, context, Z, V, mask_b, hidden
                )
                
                log_probs = F.log_softmax(logits, dim=-1)  # [beam, V]
                
                scores = beam_scores.unsqueeze(1) + log_probs  # [beam, V]
                
                if step == 0:
                    scores = scores[0]  # [V]
                    top_scores, top_indices = scores.topk(beam_size)
                    beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
                    token_indices = top_indices
                else:
                    scores_flat = scores.view(-1)  # [beam * V]
                    top_scores, top_flat_indices = scores_flat.topk(beam_size)
                    
                    beam_indices = top_flat_indices // self.vocab_size
                    token_indices = top_flat_indices % self.vocab_size
                
                new_sequences = []
                new_scores = []
                new_hidden_h = []
                new_hidden_c = []
                new_context = []
                active_beams = 0
                
                for i in range(beam_size):
                    beam_idx = beam_indices[i].item()
                    token_idx = token_indices[i].item()
                    score = top_scores[i].item()
                    
                    if token_idx == self.end_idx:
                        completed.append((score / (len(sequences[beam_idx]) + 1), sequences[beam_idx].copy()))
                    else:
                        new_seq = sequences[beam_idx] + [token_idx]
                        new_sequences.append(new_seq)
                        new_scores.append(score)
                        new_hidden_h.append(hidden[0][:, beam_idx, :])
                        new_hidden_c.append(hidden[1][:, beam_idx, :])
                        new_context.append(context[beam_idx])
                        active_beams += 1
                
                if active_beams == 0:
                    break
                
                while len(new_sequences) < beam_size:
                    new_sequences.append([])
                    new_scores.append(float('-inf'))
                    new_hidden_h.append(hidden[0][:, 0, :])
                    new_hidden_c.append(hidden[1][:, 0, :])
                    new_context.append(context[0])
                
                sequences = new_sequences[:beam_size]
                beam_scores = torch.tensor(new_scores[:beam_size], device=device)
                hidden = (
                    torch.stack(new_hidden_h[:beam_size], dim=1),
                    torch.stack(new_hidden_c[:beam_size], dim=1)
                )
                context = torch.stack(new_context[:beam_size], dim=0)
                current_tokens = torch.tensor(
                    [s[-1] if s else self.start_idx for s in sequences],
                    dtype=torch.long, device=device
                )
            
            for i, seq in enumerate(sequences):
                if seq and beam_scores[i] > float('-inf'):
                    completed.append((beam_scores[i].item() / len(seq), seq))
            
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                all_results.append(completed[0][1])
            else:
                all_results.append([])
        
        return all_results
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, strict: bool = True):
        self.load_state_dict(torch.load(path), strict=strict)

    def _encoder_summary(self, V: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        # Mean-pool encoder values over non-pad tokens
        mask = src_pad_mask.unsqueeze(-1).float()  # (B, S, 1)
        summed = (V * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom
