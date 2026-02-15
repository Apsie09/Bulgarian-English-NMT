import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

# Word-level files (Phase 1)
#corpusFileName = 'corpusData'
#wordsFileName = 'wordsData'
#modelFileName = 'NMTmodel'

# BPE files (Phase 2)
corpusFileName = 'phase2_transformer_bpe/corpusData_bpe'
wordsFileName = 'phase2_transformer_bpe/wordsData_bpe'
modelFileName = 'NMTmodel_bpe_improved_v2'

# Seq2Seq files (Phase 3)
corpusSeq2SeqFileName = 'phase3_cnn_lstm/corpusData_seq2seq_bpe'
phase3_wordsFileName = 'phase3_cnn_lstm/wordsData_seq2seq_bpe'
phase3_use_bpe = True
modelSeq2SeqFileName = 'phase3_cnn_lstm/NMTmodel_convenc_lstmdec_word_level'

# Word-level Seq2Seq (optional Phase 3 variant)
corpusSeq2SeqWordFileName = 'phase3_cnn_lstm/corpusData_seq2seq_word'
phase3_wordsFileName_word = 'phase3_cnn_lstm/wordsData_seq2seq_word'
device = torch.device("cuda:0")

# Phase 1 and 2 parameters
parameter1 = 20250  # vocabulary size
parameter2 = 4      # number of heads
parameter3 = 512    # dimension of model (must be divisible by number of heads)
parameter4 = 6     # number of layers

learning_rate = 0.0001
batchSize = 24
clip_grad = 5.0

maxEpochs = 25
log_every = 100
test_every = 2000
use_lr_scheduler = True
warmup_steps = 4000
lr_schedule_type = 'transformer'  
base_lr = 1.0  

# Phase 2 BPE settings
bpe_min_occurrence = 50


##############################################
############  Phase 3 parameters  ############
##############################################

# ConvEncoder + LSTM Decoder parameters
phase3_vocab_size = 57341      # Will be set from data
phase3_d_model = 256           # Embedding/encoder dimension
phase3_hidden_size = 512       # LSTM hidden size
phase3_enc_layers = 4        # Number of CNN layers in each stack
phase3_dec_layers = 2         # Number of LSTM layers
phase3_kernel_size = 3         # CNN kernel size
phase3_max_len = 512           # Maximum sequence length
phase3_dropout = 0.1          # Dropout rate

phase3_learning_rate = 0.0001
phase3_batch_size = 8
phase3_clip_grad = 1.0
phase3_max_epochs = 15
phase3_log_every = 100
phase3_eval_every = 2000

# Beam search parameters
phase3_beam_size = 5

# Phase 3 LR scheduling
phase3_use_lr_scheduler = True
phase3_warmup_steps = 4000
phase3_lr_schedule_type = 'transformer'
phase3_base_lr = 1.0

phase3_use_bpe = False



'''
# 1. Prepare Seq2Seq data with BPE
python run.py prepare_seq2seq_bpe 100

# 2. Run overfit test (sanity check)
python run.py train_phase3_overfit 200

# 3. Full training
python run.py train_phase3

# 4. Resume training
python run.py train_phase3 resume

# 5. Translate (greedy)
python run.py translate_phase3 en_bg_data/test.bg result_phase3.en

# 6. Translate (beam search)
python run.py translate_phase3 en_bg_data/test.bg result_phase3_beam.en beam

# 7. Evaluate BLEU
python run.py eval_phase3 en_bg_data/test.bg en_bg_data/test.en

# 8. Run shape tests
python -m models.test_shapes
'''
