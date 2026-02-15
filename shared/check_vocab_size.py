'''
import pickle
word2ind = pickle.load(open('wordsData', 'rb'))
print(f"Vocabulary size: {len(word2ind)}")
'''

'''
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
'''

import pickle
bpe = pickle.load(open('bpeModel','rb'))
print("tokens_map size:", len(bpe._tokens_map))
print("chars_map size:", len(bpe._chars_map))
word2ind = pickle.load(open('wordsData_bpe','rb'))
print("wordsData_bpe size:", len(word2ind))

