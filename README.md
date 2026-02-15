# Neural Machine Translation (BG → EN)

Course project for **Търсене и извличане на информация. Приложение на дълбоко машинно обучение** (Winter 2025/2026).

This repo contains three phases of NMT experiments:
- **Phase 1**: Decoder‑only Transformer (word‑level)
- **Phase 2**: Decoder‑only Transformer + BPE (subword)
- **Phase 3**: CNN Encoder + LSTM Decoder (seq2seq, word‑level and BPE variants)

---

## Project Structure

```
models/
  model.py                # Decoder-only Transformer (Phase 1 & 2)
  convenc_lstmdec.py      # CNN Encoder + LSTM Decoder (Phase 3)

shared/
  run.py                  # CLI: prepare / train / translate / evaluate
  parameters.py           # Hyperparameters
  utils.py                # Data preparation (word + BPE)
  tokenizer.py            # Custom BPE tokenizer

phase1_transformer/       # Saved word-level baseline
phase2_transformer_bpe/   # Saved BPE model + data
phase3_cnn_lstm/          # Saved seq2seq models
results/                  # Translation outputs
```

---

## Setup

Use the provided Conda environment:

```
conda env create -f tii.yml
conda activate tii
```

---

## Phase 1 — Transformer (word-level)

**Prepare data**
```
python shared/run.py prepare
```

**Train**
```
python shared/run.py train
```

**Translate / BLEU**
```
python shared/run.py translate en_bg_data/test.bg result_phase1.en
python shared/run.py bleu en_bg_data/test.en result_phase1.en
```

---

## Phase 2 — Transformer + BPE

**Prepare BPE data**
```
python shared/run.py prepare_bpe
```

**Train**
```
python shared/run.py train
```

**Translate (greedy)**
```
python shared/run.py translate en_bg_data/test.bg result_phase2.en
python shared/run.py bleu en_bg_data/test.en result_phase2.en
```

**Translate (beam search)**
```
python shared/run.py translate en_bg_data/test.bg result_phase2_beam.en beam
python shared/run.py bleu en_bg_data/test.en result_phase2_beam.en
```

---

## Phase 3 — CNN Encoder + LSTM Decoder (Seq2Seq)

Phase 3 supports **BPE** or **word-level** tokenization.

### Word-level (recommended)

Set in `shared/parameters.py`:
```
phase3_use_bpe = False
```

Prepare data:
```
python shared/run.py prepare_seq2seq_word [wordCountThreshold]
```

Train:
```
python shared/run.py train_phase3
```

Translate + BLEU:
```
python shared/run.py translate_phase3 en_bg_data/test.bg result_phase3_word.en beam
python shared/run.py bleu en_bg_data/test.en result_phase3_word.en
```

### BPE

Set in `shared/parameters.py`:
```
phase3_use_bpe = True
```

Prepare data:
```
python shared/run.py prepare_seq2seq_bpe [min_token_occurrence]
```

Train:
```
python shared/run.py train_phase3
```

---

## Results (current)

| Phase | Model | Tokenization | BLEU | Perplexity (dev) |
|------:|-------|--------------|------|------------------|
| 1 | Decoder-only Transformer | Word-level | 21.77 | 16.33 |
| 2 | Decoder-only Transformer | BPE | 17.86 (greedy) / 19.31 (beam=4) | ~56 |
| 3 | CNN Encoder + LSTM Decoder | Word-level | **29.79** | 11.36 |

---

## Training Parameters (summary)

**Phase 1 — Transformer (word-level)**
- d_model=256, heads=8, layers=4, d_ff=1024
- dropout=0.1, batch=8, lr=1e-4
- epochs=10

**Phase 2 — Transformer + BPE**
- d_model=512, heads=4, layers=5, d_ff=2048
- dropout=0.05, batch=24, transformer lr schedule (warmup=4000, base_lr=1.0)
- epochs=25

**Phase 3 — CNN Encoder + LSTM Decoder (word-level)**
- d_model=256, hidden=512, enc_layers=4, dec_layers=2, kernel=3
- dropout=0.1, batch=8, lr=1e-4
- epochs=15

## Notes

- Perplexity is **not directly comparable** between word-level and BPE tokenization.
- Beam search improves BLEU but is slower.
- Phase 3 performs best with **word-level** data (BPE underperformed for CNN+LSTM).

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Bahdanau et al. (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al. (2015) — *Effective Approaches to Attention-based NMT*
- Gehring et al. (2016) — *Convolutional Sequence to Sequence Learning*

---

If you want to reproduce a specific run, check `shared/parameters.py` for the exact hyperparameters.
