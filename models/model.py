#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch

class PositionalEncoding(torch.nn.Module):

	def __init__(self, d_model, max_len=5000):
		super().__init__()
		
		pe = torch.zeros(1, max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe[0,:, 0::2] = torch.sin(position * div_term)
		pe[0,:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1), :]
		return x

class MultiHeadAttention(torch.nn.Module):

	def __init__(self, num_heads, d_model, d_keys, d_values):
		super(MultiHeadAttention, self).__init__()
		assert d_model % num_heads == 0 #d_model must be divisible by num_heads

		self.num_heads, self.d_model, self.d_keys, self.d_values = num_heads, d_model, d_keys, d_values
		self.scale = 1 / (d_keys ** 0.5) 

		self.W_q = torch.nn.Linear(d_model, num_heads * d_keys)
		self.W_k = torch.nn.Linear(d_model, num_heads * d_keys)
		self.W_v = torch.nn.Linear(d_model, num_heads * d_values)
		self.W_o = torch.nn.Linear(num_heads * d_values, d_model)
	
	def forward(self, input_X, 
			 mask = None):
		num_heads, d_model, d_keys, d_values = self.num_heads, self.d_model, self.d_keys, self.d_values
		# input_X shape: (batch_size, seq_len, d_model)
		batch_size = input_X.shape[0] 
		seq_len = input_X.shape[1]

		head_q = self.W_q(input_X) # shape: (batch_size, seq_len, num_heads * d_keys)
		head_k = self.W_k(input_X) # shape: (batch_size, seq_len, num_heads * d_keys)
		head_v = self.W_v(input_X) # shape: (batch_size, seq_len, num_heads * d_values)

		q = head_q.view(batch_size, seq_len, num_heads, d_keys).transpose(1,2) # shape: (batch_size, num_heads, seq_len, d_keys)
		k = head_k.view(batch_size, seq_len, num_heads, d_keys).transpose(1,2) # shape: (batch_size, num_heads, seq_len, d_keys)
		v = head_v.view(batch_size, seq_len, num_heads, d_values).transpose(1,2) # shape: (batch_size, num_heads, seq_len, d_values)

		attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # shape: (batch_size, num_heads, seq_len, seq_len)

		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

		attn_probs = torch.nn.functional.softmax(attn_scores, dim=3) # shape: (batch_size, num_heads, seq_len, seq_len)
		att_vec = torch.matmul(attn_probs, v) # shape: (batch_size, num_heads, seq_len, d_values)
		att_vec = att_vec.transpose(1,2).flatten(2,3) # shape: (batch_size, seq_len, num_heads * d_values)

		output = self.W_o(att_vec) # shape: (batch_size, seq_len, d_model)
		return output

class TransformerBlock(torch.nn.Module):

	def __init__(self, num_heads, d_model, d_keys, d_values, d_ff, dropout):

		super().__init__()
		self.MHA = MultiHeadAttention(num_heads, d_model, d_keys, d_values)
		self.layer_norm1 = torch.nn.LayerNorm(d_model)
		self.dropout1 = torch.nn.Dropout(dropout)
		self.ff1 = torch.nn.Linear(d_model, d_ff)
		self.ff2 = torch.nn.Linear(d_ff, d_model)
		self.layer_norm2 = torch.nn.LayerNorm(d_model)
		self.dropout2 = torch.nn.Dropout(dropout)

	def forward(self, input_X, mask = None):
		# input_X shape: (batch_size, seq_len, d_model)
		# mask shape: (batch_size, seq_len, seq_len)

		attn_output = self.MHA(input_X, mask)
		out1 = self.layer_norm1(input_X + self.dropout1(attn_output))

		# Feed Forward Network with Residual Connection and Layer Norm
		ff_output = self.ff2(torch.nn.functional.relu(self.ff1(out1)))
		out2 = self.layer_norm2(out1 + self.dropout2(ff_output)) # Output - Y
		
		return out2


# Transformer-based Language Model

class LanguageModel(torch.nn.Module):
	def __init__(self, vocab_size, num_heads, d_model, num_layers, max_len=5000):
		super(LanguageModel, self).__init__()

		self.d_model = d_model
		self.vocab_size = vocab_size

		# Special Token Indices
		self.startTokenIdx = 0
		self.endTokenIdx = 1
		self.unknownTokenIdx = 2
		self.padTokenIdx = 3
		self.transTokenIdx = 4

		# Embedding Layer
		self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=self.padTokenIdx)

		# Positional Encoding
		self.positional_encoding = PositionalEncoding(d_model, max_len)
		
		# Dropout
		self.dropout = torch.nn.Dropout(0.05)

		# Transformer Blocks
		self.Transformer = torch.nn.ModuleList([
			TransformerBlock(num_heads, d_model, d_model // num_heads, d_model // num_heads, d_model * 4, 0.05)
			for _ in range(num_layers)
		])

		# Output Layer
		self.output_layer = torch.nn.Linear(d_model, vocab_size)

		# Mask for Causal Attention
		pos = torch.arange(max_len)
		mask = pos.unsqueeze(0) <= pos.unsqueeze(1)
		self.register_buffer('mask', mask)

		# Initialize weights
		self._init_weights()
	
	# Xavier Initialization to prevent vanishing/exploding gradients
	def _init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				torch.nn.init.xavier_uniform_(p)
	
	def preparePaddedBatch(self, source):
		device = next(self.parameters()).device
		m = max(len(s) for s in source)
		sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in source]
		return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

	def save(self,fileName):
		torch.save(self.state_dict(), fileName)

	def load(self,fileName):
		self.load_state_dict(torch.load(fileName))

	def forward(self, source):
		X = self.preparePaddedBatch(source)  # shape=(batch_size, seq_len)
		seq_len = X.shape[1]
		Emb_X = self.embedding(X[:, :-1])  # shape=(batch_size, seq_len-1, d_model)
		input = self.dropout(self.positional_encoding(Emb_X))

		causal_mask = self.mask[:seq_len-1, :seq_len-1] # shape=(seq_len-1, seq_len-1)
		pad_mask = (X[:, :-1] != self.padTokenIdx).unsqueeze(1) # shape=(batch_size, 1, seq_len-1)
		batch_mask = causal_mask.unsqueeze(0) & pad_mask # shape=(batch_size, seq_len-1, seq_len-1)
		hidden = input
		for block in self.Transformer:
			hidden = block(hidden, batch_mask)

		Z = self.output_layer(hidden.flatten(0,1))
		Y_bar = X[:, 1:].flatten(0,1)
		H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.padTokenIdx)
		return H
	
	@torch.no_grad()
	def generate(self, prefix, limit=1000):
		self.eval()
		device = next(self.parameters()).device

		# Start with the prefix
		result = list(prefix)

		with torch.no_grad():
			for _ in range(limit):
				X = torch.tensor([result], dtype=torch.long, device=device)  # shape=(1, current_seq_len)
				seq_len = X.shape[1]

				Emb_X = self.embedding(X)
				input = self.positional_encoding(Emb_X)

				causal_mask = self.mask[:seq_len, :seq_len]
				pad_mask = (X != self.padTokenIdx).unsqueeze(1)
				batch_mask = causal_mask.unsqueeze(0) & pad_mask

				hidden = input
				for block in self.Transformer:
					hidden = block(hidden, batch_mask)

				# last token
				Z = self.output_layer(hidden[:, -1, :])
				# remove unknown, padding, and trans tokens from the vocabulary
				Z[:, self.unknownTokenIdx] = float('-inf')
				Z[:, self.padTokenIdx] = float('-inf')
				Z[:, self.transTokenIdx] = float('-inf')

				next_token = torch.argmax(Z, dim=-1).item()
				
				result.append(next_token)

				if next_token == self.endTokenIdx:
					break
			
		return result

	@torch.no_grad()
	def beam_search(self, prefix, beam_size=4, limit=1000, length_norm=0.7):
		self.eval()
		device = next(self.parameters()).device

		def score_fn(logprob, length):
			return logprob / (length ** length_norm)

		# (sequence, logprob_sum)
		beams = [(list(prefix), 0.0)]
		completed = []

		for _ in range(limit):
			new_beams = []
			for seq, logprob in beams:
				if seq and seq[-1] == self.endTokenIdx:
					completed.append((seq, logprob))
					continue

				X = torch.tensor([seq], dtype=torch.long, device=device)
				seq_len = X.shape[1]

				Emb_X = self.embedding(X)
				input = self.positional_encoding(Emb_X)

				causal_mask = self.mask[:seq_len, :seq_len]
				pad_mask = (X != self.padTokenIdx).unsqueeze(1)
				batch_mask = causal_mask.unsqueeze(0) & pad_mask

				hidden = input
				for block in self.Transformer:
					hidden = block(hidden, batch_mask)

				logits = self.output_layer(hidden[:, -1, :])
				logits[:, self.unknownTokenIdx] = float('-inf')
				logits[:, self.padTokenIdx] = float('-inf')
				logits[:, self.transTokenIdx] = float('-inf')

				log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
				top_log_probs, top_indices = torch.topk(log_probs, k=beam_size)

				for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
					new_seq = seq + [idx]
					new_beams.append((new_seq, logprob + lp))

			if not new_beams:
				break

			# select top beams by length-normalized score
			new_beams.sort(key=lambda x: score_fn(x[1], len(x[0])), reverse=True)
			beams = new_beams[:beam_size]

			if len(completed) >= beam_size:
				break

		if completed:
			completed.sort(key=lambda x: score_fn(x[1], len(x[0])), reverse=True)
			return completed[0][0]

		return beams[0][0]
