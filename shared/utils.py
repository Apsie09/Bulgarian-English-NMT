#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')

from tokenizer import Tokenizer

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    return [ nltk.word_tokenize(line) for line in open(fileName) ]

def getDictionary(corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold = 2):
    dictionary={}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w]=1

    words = [startToken, endToken, unkToken, padToken, transToken] + [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    return { w:i for i,w in enumerate(words)}


def prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    word2ind = getDictionary(sourceCorpus+targetCorpus, startToken, endToken, unkToken, padToken, transToken)

    trainCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceCorpus,targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceDev,targetDev)]

    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind

# Read Corupus as Raw Strings for BPE Tokenizer Training
def readCorpusRaw(fileName):
    print('Loading file:', fileName)
    return [line.strip() for line in open(fileName)]

def prepareDataBPE(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, 
                   startToken, endToken, unkToken, padToken, transToken, min_token_occurrence=100):
    
    sourceCorpusRaw = readCorpusRaw(sourceFileName)
    targetCorpusRaw = readCorpusRaw(targetFileName)
    
    combined = sourceCorpusRaw + targetCorpusRaw
    bpe = Tokenizer(min_token_occurrence)
    bpe.train(combined)
    
    # Build remapped vocabulary with special tokens at 0-4
    special_tokens = [startToken, endToken, unkToken, padToken, transToken]  # indices 0,1,2,3,4
    
    word2ind = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add all BPE tokens, shifting their IDs to start at 5
    next_id = 5
    old_to_new = {}  
    
    for old_id, token_str in bpe._tokens_map.items():
        if token_str in word2ind:
            old_to_new[old_id] = word2ind[token_str]
        else:
            word2ind[token_str] = next_id
            old_to_new[old_id] = next_id
            next_id += 1
    
    ind2word = {v: k for k, v in word2ind.items()}
    
    print(f"Vocabulary size: {len(word2ind)}")
    print(f"Special tokens: <S>={word2ind[startToken]}, </S>={word2ind[endToken]}, <UNK>={word2ind[unkToken]}, <PAD>={word2ind[padToken]}, <TRANS>={word2ind[transToken]}")
    
    bpe._old_to_new = old_to_new
    
    print("Tokenizing training corpus...")
    trainCorpus = []
    for src, tgt in zip(sourceCorpusRaw, targetCorpusRaw):
        src_tokens = [old_to_new[t] for t in bpe.to_tokens(src)]
        tgt_tokens = [old_to_new[t] for t in bpe.to_tokens(tgt)]
        seq = [word2ind[startToken]] + src_tokens + [word2ind[transToken]] + tgt_tokens + [word2ind[endToken]]
        trainCorpus.append(seq)
    
    sourceDevRaw = readCorpusRaw(sourceDevFileName)
    targetDevRaw = readCorpusRaw(targetDevFileName)
    
    print("Tokenizing dev corpus...")
    devCorpus = []
    for src, tgt in zip(sourceDevRaw, targetDevRaw):
        src_tokens = [old_to_new[t] for t in bpe.to_tokens(src)]
        tgt_tokens = [old_to_new[t] for t in bpe.to_tokens(tgt)]
        seq = [word2ind[startToken]] + src_tokens + [word2ind[transToken]] + tgt_tokens + [word2ind[endToken]]
        devCorpus.append(seq)
    
    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind, bpe

def prepareDataSeq2SeqBPE(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName,
                          startToken, endToken, unkToken, padToken, transToken, 
                          min_token_occurrence=100):

    sourceCorpusRaw = readCorpusRaw(sourceFileName)
    targetCorpusRaw = readCorpusRaw(targetFileName)
    
    combined = sourceCorpusRaw + targetCorpusRaw
    bpe = Tokenizer(min_token_occurrence)
    bpe.train(combined)
    
    special_tokens = [startToken, endToken, unkToken, padToken, transToken]
    word2ind = {token: idx for idx, token in enumerate(special_tokens)}
    
    next_id = 5
    old_to_new = {}
    
    for old_id, token_str in bpe._tokens_map.items():
        if token_str in word2ind:
            old_to_new[old_id] = word2ind[token_str]
        else:
            word2ind[token_str] = next_id
            old_to_new[old_id] = next_id
            next_id += 1
    
    print(f"Vocabulary size: {len(word2ind)}")
    print(f"Special tokens: <S>={word2ind[startToken]}, </S>={word2ind[endToken]}, "
          f"<UNK>={word2ind[unkToken]}, <PAD>={word2ind[padToken]}, <TRANS>={word2ind[transToken]}")
    
    bpe._old_to_new = old_to_new
    
    print("Tokenizing training corpus for Seq2Seq...")
    trainSrc = []
    trainTgt = []
    for src, tgt in zip(sourceCorpusRaw, targetCorpusRaw):
        src_tokens = [old_to_new[t] for t in bpe.to_tokens(src)]
        tgt_tokens = [old_to_new[t] for t in bpe.to_tokens(tgt)]
        trainSrc.append(src_tokens)
        trainTgt.append(tgt_tokens)
    
    sourceDevRaw = readCorpusRaw(sourceDevFileName)
    targetDevRaw = readCorpusRaw(targetDevFileName)
    
    print("Tokenizing dev corpus for Seq2Seq...")
    devSrc = []
    devTgt = []
    for src, tgt in zip(sourceDevRaw, targetDevRaw):
        src_tokens = [old_to_new[t] for t in bpe.to_tokens(src)]
        tgt_tokens = [old_to_new[t] for t in bpe.to_tokens(tgt)]
        devSrc.append(src_tokens)
        devTgt.append(tgt_tokens)
    
    print('Seq2Seq corpus loading completed.')
    return trainSrc, trainTgt, devSrc, devTgt, word2ind, bpe


def prepareDataSeq2SeqWord(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName,
                           startToken, endToken, unkToken, padToken, transToken,
                           wordCountThreshold=2):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)

    # Shared vocab across source + target (same as prepareData)
    word2ind = getDictionary(
        sourceCorpus + targetCorpus,
        startToken, endToken, unkToken, padToken, transToken,
        wordCountThreshold=wordCountThreshold
    )

    # Convert to ids
    trainSrc = [[word2ind.get(w, word2ind[unkToken]) for w in s] for s in sourceCorpus]
    trainTgt = [[word2ind.get(w, word2ind[unkToken]) for w in s] for s in targetCorpus]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devSrc = [[word2ind.get(w, word2ind[unkToken]) for w in s] for s in sourceDev]
    devTgt = [[word2ind.get(w, word2ind[unkToken]) for w in s] for s in targetDev]

    print('Seq2Seq word-level corpus loading completed.')
    return trainSrc, trainTgt, devSrc, devTgt, word2ind


def compute_bleu_score(references: list, hypotheses: list) -> float:
    
    refs = [[ref] for ref in references]
    return corpus_bleu(refs, hypotheses) * 100
