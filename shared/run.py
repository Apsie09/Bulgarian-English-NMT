#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import math
import pickle
import time
import nltk

from nltk.translate.bleu_score import corpus_bleu
from models.convenc_lstmdec import ConvEncoderLSTMDecoder
from models.seq2seq_data import create_seq2seq_dataloader

import utils
import models.model as model
from parameters import *

from parameters import phase3_use_lr_scheduler, phase3_warmup_steps, phase3_lr_schedule_type, phase3_base_lr
from parameters import phase3_wordsFileName, phase3_use_bpe, corpusSeq2SeqWordFileName, phase3_wordsFileName_word


startToken = '<S>'
startTokenIdx = 0

endToken = '</S>'
endTokenIdx = 1

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'
padTokenIdx = 3

transToken = '<TRANS>'
transTokenIdx = 4

# Add after: from parameters import *

def get_transformer_lr_schedule(d_model, warmup_steps=4000):

    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda

def get_warmup_cosine_schedule(warmup_steps, total_steps):

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def perplexity(nmt, test, batchSize):
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        batch = test[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H/c)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    trainCorpus, devCorpus, word2ind = utils.prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken)
    trainCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in trainCorpus ]
    devCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in devCorpus ]
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and sys.argv[1] == 'prepare_bpe':
    min_occurrence = bpe_min_occurrence
    if len(sys.argv) > 2:
        min_occurrence = int(sys.argv[2])
    
    trainCorpus, devCorpus, word2ind, bpe = utils.prepareDataBPE(
        sourceFileName, targetFileName, 
        sourceDevFileName, targetDevFileName, 
        startToken, endToken, unkToken, padToken, transToken,
        min_token_occurrence=min_occurrence
    )
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    pickle.dump(bpe, open('bpeModel', 'wb'))
    print(f'Data prepared with BPE. Vocabulary size: {len(word2ind)}')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (trainCorpus,devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(parameter1, parameter2, parameter3, parameter4).to(device)
    if use_lr_scheduler and lr_schedule_type == 'transformer':

        optimizer = torch.optim.Adam(nmt.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
    else:

        optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    scheduler = None
    if use_lr_scheduler:
        if lr_schedule_type == 'transformer':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=get_transformer_lr_schedule(d_model=parameter3, warmup_steps=warmup_steps)
            )
        elif lr_schedule_type == 'cosine':
            total_steps = maxEpochs * (len(trainCorpus) // batchSize)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=get_warmup_cosine_schedule(warmup_steps=warmup_steps, total_steps=total_steps)
            )
    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
			#############################################################################
			### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
			#############################################################################
            iter += 1
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            
            words += sum( len(s)-1 for s in batch )
            H = nmt(batch)
            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if iter % log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,
                      ", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, 
                      ", loss: ",H.item(), 
                      ", LR: {:.6f}".format(current_lr),
                      "words/sec:",words / (time.time() - trainTime), 
                      "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    
    nmt = model.LanguageModel(parameter1, parameter2, parameter3, parameter4).to(device)
    nmt.load(modelFileName)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    test = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceTest,targetTest)]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = {v: k for k, v in word2ind.items()}

    try:
        bpe = pickle.load(open('bpeModel', 'rb'))
        use_bpe = True
        print("Using BPE tokenization")
    except FileNotFoundError:
        use_bpe = False
        print("Using word-level tokenization")

    nmt = model.LanguageModel(parameter1, parameter2, parameter3, parameter4).to(device)
    nmt.load(modelFileName)
    nmt.eval()

    use_beam = len(sys.argv) > 4 and sys.argv[4] == 'beam'

    if use_bpe:
        sourceTestRaw = utils.readCorpusRaw(sys.argv[2])
        old_to_new = getattr(bpe, '_old_to_new', None)  
        test = []
        for src in sourceTestRaw:
            src_tokens = bpe.to_tokens(src)
            if old_to_new:
                src_tokens = [old_to_new[t] for t in src_tokens] 
            seq = [word2ind[startToken]] + src_tokens + [word2ind[transToken]]
            test.append(seq)
    else:
        sourceTest = utils.readCorpus(sys.argv[2])
        test = [[startToken] + s + [transToken] for s in sourceTest]
        test = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test]

    file = open(sys.argv[3], 'w')
    pb = utils.progressBar()
    pb.start(len(test))
    
    for s in test:
        if use_beam:
            r = nmt.beam_search(s, beam_size=4)
        else:
            r = nmt.generate(s)
        st = r.index(word2ind[transToken])
        result_ids = r[st+1:-1]
        
        if use_bpe:
            result_text = ''.join(words.get(tid, '') for tid in result_ids)
        else:
            result_text = ' '.join(words.get(tid, '') for tid in result_ids)
        
        file.write(result_text.strip() + "\n")
        pb.tick()
    
    pb.stop()
    file.close()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    test = sys.argv[2].split()
    test = [word2ind.get(w,unkTokenIdx) for w in test]

    nmt = model.LanguageModel(parameter1, parameter2, parameter3, parameter4).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    r=nmt.generate(test)
    result = [words[i] for i in r]
    print(' '.join(result)+"\n")

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))

# #########################################
# PHASE 3: ConvEncoder + LSTM Decoder
# #########################################

if len(sys.argv) > 1 and sys.argv[1] == 'prepare_seq2seq_bpe':
    
    min_occurrence = 100
    if len(sys.argv) > 2:
        min_occurrence = int(sys.argv[2])
    
    trainSrc, trainTgt, devSrc, devTgt, word2ind, bpe = utils.prepareDataSeq2SeqBPE(
        sourceFileName, targetFileName,
        sourceDevFileName, targetDevFileName,
        startToken, endToken, unkToken, padToken, transToken,
        min_token_occurrence=min_occurrence
    )
    
    pickle.dump((trainSrc, trainTgt, devSrc, devTgt), open(corpusSeq2SeqFileName, 'wb'))
    pickle.dump(word2ind, open(phase3_wordsFileName, 'wb'))
    pickle.dump(bpe, open(ROOT / 'bpeModel', 'wb'))
    print(f'Seq2Seq data prepared with BPE. Vocabulary size: {len(word2ind)}')

if len(sys.argv) > 1 and sys.argv[1] == 'prepare_seq2seq_word':

    wordCountThreshold = 2
    if len(sys.argv) > 2:
        wordCountThreshold = int(sys.argv[2])

    trainSrc, trainTgt, devSrc, devTgt, word2ind = utils.prepareDataSeq2SeqWord(
        sourceFileName, targetFileName,
        sourceDevFileName, targetDevFileName,
        startToken, endToken, unkToken, padToken, transToken,
        wordCountThreshold=wordCountThreshold
    )

    pickle.dump((trainSrc, trainTgt, devSrc, devTgt), open(corpusSeq2SeqWordFileName, 'wb'))
    pickle.dump(word2ind, open(phase3_wordsFileName_word, 'wb'))
    print(f'Seq2Seq data prepared (word-level). Vocabulary size: {len(word2ind)}')


if len(sys.argv) > 1 and sys.argv[1] == 'train_phase3':
    
    from parameters import (phase3_d_model, phase3_hidden_size, phase3_enc_layers,
                           phase3_dec_layers, phase3_kernel_size, phase3_max_len,
                           phase3_dropout, phase3_learning_rate, phase3_batch_size,
                           phase3_clip_grad, phase3_max_epochs, phase3_log_every,
                           phase3_eval_every, corpusSeq2SeqFileName, modelSeq2SeqFileName)
    
    if phase3_use_bpe:
        (trainSrc, trainTgt, devSrc, devTgt) = pickle.load(open(corpusSeq2SeqFileName, 'rb'))
        word2ind = pickle.load(open(phase3_wordsFileName, 'rb'))
    else:
        (trainSrc, trainTgt, devSrc, devTgt) = pickle.load(open(corpusSeq2SeqWordFileName, 'rb'))
        word2ind = pickle.load(open(phase3_wordsFileName_word, 'rb'))
    vocab_size = len(word2ind)
    
    print(f"Training data: {len(trainSrc)} pairs")
    print(f"Dev data: {len(devSrc)} pairs")
    print(f"Vocabulary size: {vocab_size}")
    
    model = ConvEncoderLSTMDecoder(
        vocab_size=vocab_size,
        d_model=phase3_d_model,
        hidden_size=phase3_hidden_size,
        enc_layers=phase3_enc_layers,
        dec_layers=phase3_dec_layers,
        kernel_size=phase3_kernel_size,
        max_len=phase3_max_len,
        dropout=phase3_dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if phase3_use_lr_scheduler:
        optimizer = torch.optim.Adam(model.parameters(), lr=phase3_base_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=get_transformer_lr_schedule(d_model=phase3_d_model, warmup_steps=phase3_warmup_steps)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=phase3_learning_rate)
        scheduler = None

    best_ppl = math.inf
    start_epoch = 0
    
    if len(sys.argv) > 2 and sys.argv[2] == 'resume':
        try:
            model.load(modelSeq2SeqFileName)
            checkpoint = torch.load(modelSeq2SeqFileName + '.optim')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_ppl = checkpoint['best_ppl']
            print(f"Resuming from epoch {start_epoch}, best ppl: {best_ppl:.2f}")
        except FileNotFoundError:
            print("No checkpoint found, starting fresh.")
    
    train_loader = create_seq2seq_dataloader(
        trainSrc, trainTgt, phase3_batch_size,
        start_idx=startTokenIdx, end_idx=endTokenIdx, pad_idx=padTokenIdx,
        shuffle=True
    )
    
    dev_loader = create_seq2seq_dataloader(
        devSrc, devTgt, phase3_batch_size,
        start_idx=startTokenIdx, end_idx=endTokenIdx, pad_idx=padTokenIdx,
        shuffle=False
    )
    
    global_step = 0
    begin_time = time.time()
    
    for epoch in range(start_epoch, phase3_max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        train_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            
            src_ids = batch['src_ids'].to(device)
            tgt_in_ids = batch['tgt_in_ids'].to(device)
            tgt_out_ids = batch['tgt_out_ids'].to(device)
            src_pad_mask = batch['src_pad_mask'].to(device)
            tgt_pad_mask = batch['tgt_pad_mask'].to(device)
            
            loss = model.compute_loss(src_ids, tgt_in_ids, tgt_out_ids, src_pad_mask, tgt_pad_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), phase3_clip_grad)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            num_tokens = tgt_pad_mask.sum().item()
            epoch_loss += loss.item() * num_tokens
            epoch_tokens += num_tokens
            
            if global_step % phase3_log_every == 0:
                avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
                elapsed = time.time() - begin_time
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{phase3_max_epochs}, Step {global_step}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, "
                      f"Time: {elapsed:.1f}s, LR: {current_lr:.6f}")
            
            if global_step % phase3_eval_every == 0:
                model.eval()
                
                dev_loss = 0.0
                dev_tokens = 0
                
                with torch.no_grad():
                    for dev_batch in dev_loader:
                        src_ids = dev_batch['src_ids'].to(device)
                        tgt_in_ids = dev_batch['tgt_in_ids'].to(device)
                        tgt_out_ids = dev_batch['tgt_out_ids'].to(device)
                        src_pad_mask = dev_batch['src_pad_mask'].to(device)
                        tgt_pad_mask = dev_batch['tgt_pad_mask'].to(device)
                        
                        loss = model.compute_loss(src_ids, tgt_in_ids, tgt_out_ids, 
                                                  src_pad_mask, tgt_pad_mask)
                        num_tokens = tgt_pad_mask.sum().item()
                        dev_loss += loss.item() * num_tokens
                        dev_tokens += num_tokens
                
                dev_ppl = math.exp(dev_loss / dev_tokens) if dev_tokens > 0 else float('inf')
                print(f"Dev Perplexity: {dev_ppl:.2f}")
                
                if dev_ppl < best_ppl:
                    best_ppl = dev_ppl
                    print(f"New best model! Saving...")
                    model.save(modelSeq2SeqFileName)
                    torch.save({
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'best_ppl': best_ppl,
                        'global_step': global_step
                    }, modelSeq2SeqFileName + '.optim')
                
                model.train()
        
        avg_epoch_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
    
    print(f"Training completed! Best dev perplexity: {best_ppl:.2f}")


if len(sys.argv) > 1 and sys.argv[1] == 'train_phase3_overfit':
    
    from parameters import (phase3_d_model, phase3_hidden_size, phase3_enc_layers,
                           phase3_dec_layers, phase3_kernel_size, phase3_max_len,
                           phase3_dropout, phase3_learning_rate, phase3_batch_size,
                           phase3_clip_grad, corpusSeq2SeqFileName, modelSeq2SeqFileName)
    
    num_samples = 20
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    
    if phase3_use_bpe:
        (trainSrc, trainTgt, devSrc, devTgt) = pickle.load(open(corpusSeq2SeqFileName, 'rb'))
        word2ind = pickle.load(open(phase3_wordsFileName, 'rb'))
    else:
        (trainSrc, trainTgt, devSrc, devTgt) = pickle.load(open(corpusSeq2SeqWordFileName, 'rb'))
        word2ind = pickle.load(open(phase3_wordsFileName_word, 'rb'))
    ind2word = {v: k for k, v in word2ind.items()}
    vocab_size = len(word2ind)
    
    trainSrc = trainSrc[:num_samples]
    trainTgt = trainTgt[:num_samples]
    
    print(f"Overfit test with {num_samples} samples")
    print(f"Vocabulary size: {vocab_size}")
    
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
    
    # Stronger LR for memorization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader = create_seq2seq_dataloader(
        trainSrc, trainTgt, min(phase3_batch_size, num_samples),
        start_idx=startTokenIdx, end_idx=endTokenIdx, pad_idx=padTokenIdx,
        shuffle=True
    )
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            src_ids = batch['src_ids'].to(device)
            tgt_in_ids = batch['tgt_in_ids'].to(device)
            tgt_out_ids = batch['tgt_out_ids'].to(device)
            src_pad_mask = batch['src_pad_mask'].to(device)
            tgt_pad_mask = batch['tgt_pad_mask'].to(device)
            
            loss = model.compute_loss(src_ids, tgt_in_ids, tgt_out_ids, src_pad_mask, tgt_pad_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), phase3_clip_grad)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    for i in range(min(3, num_samples)):
                        src = [startTokenIdx] + trainSrc[i] + [endTokenIdx]
                        src_tensor = torch.tensor([src], device=device)
                        src_mask = torch.ones(1, len(src), dtype=torch.bool, device=device)
                        
                        decoded = model.greedy_decode(src_tensor, src_mask, max_len=50)
                        
                        # Convert to text
                        src_text = ''.join(ind2word.get(t, '?') for t in trainSrc[i])
                        ref_text = ''.join(ind2word.get(t, '?') for t in trainTgt[i])
                        hyp_text = ''.join(ind2word.get(t, '?') for t in decoded[0])
                        
                        print(f"\n  Src: {src_text[:80]}...")
                        print(f"  Ref: {ref_text[:80]}...")
                        print(f"  Hyp: {hyp_text[:80]}...")
    
    print("\nOverfit test completed!")


if len(sys.argv) > 3 and sys.argv[1] == 'translate_phase3':
    
    from parameters import (phase3_d_model, phase3_hidden_size, phase3_enc_layers,
                           phase3_dec_layers, phase3_kernel_size, phase3_max_len,
                           phase3_dropout, phase3_beam_size, modelSeq2SeqFileName)
    
    if phase3_use_bpe:
        words_path = ROOT / phase3_wordsFileName
    else:
        words_path = ROOT / phase3_wordsFileName_word
    word2ind = pickle.load(open(words_path, 'rb'))
    ind2word = {v: k for k, v in word2ind.items()}
    vocab_size = len(word2ind)
    
    if phase3_use_bpe:
        _bpe_candidates = [
            ROOT / 'phase2_transformer_bpe' / 'bpeModel',
            ROOT / 'bpeModel',
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
    try:
        model.load(modelSeq2SeqFileName)
    except RuntimeError:
        model.use_enc_init = False
        model.load(modelSeq2SeqFileName, strict=False)
    model.eval()
    
    use_beam = len(sys.argv) > 4 and sys.argv[4] == 'beam'
    
    source_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if phase3_use_bpe:
        sourceRaw = utils.readCorpusRaw(source_file)
    else:
        sourceRaw = utils.readCorpus(source_file)
    
    print(f"Translating {len(sourceRaw)} sentences...")
    print(f"Using {'beam search' if use_beam else 'greedy'} decoding")
    
    results = []
    pb = utils.progressBar()
    pb.start(len(sourceRaw))
    
    with torch.no_grad():
        for src_text in sourceRaw:
            if phase3_use_bpe:
                src_tokens = [old_to_new[t] for t in bpe.to_tokens(src_text)]
            else:
                src_tokens = [word2ind.get(w, unkTokenIdx) for w in src_text]
            src_ids = [startTokenIdx] + src_tokens + [endTokenIdx]
            
            src_tensor = torch.tensor([src_ids], device=device)
            src_mask = torch.ones(1, len(src_ids), dtype=torch.bool, device=device)
            
            if use_beam:
                decoded = model.beam_search(src_tensor, src_mask, beam_size=phase3_beam_size)
            else:
                decoded = model.greedy_decode(src_tensor, src_mask)
            
            if phase3_use_bpe:
                result_text = ''.join(ind2word.get(t, '') for t in decoded[0])
            else:
                result_text = ' '.join(ind2word.get(t, '') for t in decoded[0])
            results.append(result_text)
            pb.tick()
    
    pb.stop()
    
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line.strip() + '\n')
    
    print(f"Translations written to {output_file}")


if len(sys.argv) > 3 and sys.argv[1] == 'eval_phase3':

    from parameters import (phase3_d_model, phase3_hidden_size, phase3_enc_layers,
                           phase3_dec_layers, phase3_kernel_size, phase3_max_len,
                           phase3_beam_size, modelSeq2SeqFileName)
    
    if phase3_use_bpe:
        words_path = ROOT / phase3_wordsFileName
    else:
        words_path = ROOT / phase3_wordsFileName_word
    word2ind = pickle.load(open(words_path, 'rb'))
    ind2word = {v: k for k, v in word2ind.items()}
    vocab_size = len(word2ind)
    
    if phase3_use_bpe:
        _bpe_candidates = [
            ROOT / 'phase2_transformer_bpe' / 'bpeModel',
            ROOT / 'bpeModel',
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
    try:
        model.load(modelSeq2SeqFileName)
    except RuntimeError:
        model.use_enc_init = False
        model.load(modelSeq2SeqFileName, strict=False)
    model.eval()
    
    source_file = sys.argv[2]
    reference_file = sys.argv[3]
    
    if phase3_use_bpe:
        sourceRaw = utils.readCorpusRaw(source_file)
    else:
        sourceRaw = utils.readCorpus(source_file)
    referenceCorpus = utils.readCorpus(reference_file)  
    
    hypotheses = []
    pb = utils.progressBar()
    pb.start(len(sourceRaw))
    
    with torch.no_grad():
        for src_text in sourceRaw:
            if phase3_use_bpe:
                src_tokens = [old_to_new[t] for t in bpe.to_tokens(src_text)]
            else:
                src_tokens = [word2ind.get(w, unkTokenIdx) for w in src_text]
            src_ids = [startTokenIdx] + src_tokens + [endTokenIdx]
            
            src_tensor = torch.tensor([src_ids], device=device)
            src_mask = torch.ones(1, len(src_ids), dtype=torch.bool, device=device)
            
            decoded = model.greedy_decode(src_tensor, src_mask)
            if phase3_use_bpe:
                result_text = ''.join(ind2word.get(t, '') for t in decoded[0])
            else:
                result_text = ' '.join(ind2word.get(t, '') for t in decoded[0])
            
            hyp_tokens = nltk.word_tokenize(result_text)
            hypotheses.append(hyp_tokens)
            pb.tick()
    
    pb.stop()
    
    bleu = utils.compute_bleu_score(referenceCorpus, hypotheses)
    print(f"BLEU Score: {bleu:.2f}")
