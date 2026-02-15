import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


class Seq2SeqDataset(Dataset):
    def __init__(self, src_sequences: List[List[int]], tgt_sequences: List[List[int]],
                 start_idx: int = 0, end_idx: int = 1):

        assert len(src_sequences) == len(tgt_sequences), "Source and target must have same length"
        
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def __len__(self) -> int:
        return len(self.src_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        src = self.src_sequences[idx]
        tgt = self.tgt_sequences[idx]
        
        src_ids = [self.start_idx] + src + [self.end_idx]
        tgt_in_ids = [self.start_idx] + tgt
        tgt_out_ids = tgt + [self.end_idx]
        
        return src_ids, tgt_in_ids, tgt_out_ids


class Seq2SeqCollator:
    def __init__(self, pad_idx: int = 3):
        self.pad_idx = pad_idx
    
    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:

        src_seqs, tgt_in_seqs, tgt_out_seqs = zip(*batch)
        
        max_src_len = max(len(s) for s in src_seqs)
        max_tgt_len = max(len(s) for s in tgt_in_seqs)
        
        batch_size = len(batch)
        
        src_ids = torch.full((batch_size, max_src_len), self.pad_idx, dtype=torch.long)
        tgt_in_ids = torch.full((batch_size, max_tgt_len), self.pad_idx, dtype=torch.long)
        tgt_out_ids = torch.full((batch_size, max_tgt_len), self.pad_idx, dtype=torch.long)
        
        for i, (src, tgt_in, tgt_out) in enumerate(batch):
            src_ids[i, :len(src)] = torch.tensor(src, dtype=torch.long)
            tgt_in_ids[i, :len(tgt_in)] = torch.tensor(tgt_in, dtype=torch.long)
            tgt_out_ids[i, :len(tgt_out)] = torch.tensor(tgt_out, dtype=torch.long)
        
        src_pad_mask = src_ids != self.pad_idx
        tgt_pad_mask = tgt_in_ids != self.pad_idx
        
        return {
            'src_ids': src_ids,
            'tgt_in_ids': tgt_in_ids,
            'tgt_out_ids': tgt_out_ids,
            'src_pad_mask': src_pad_mask,
            'tgt_pad_mask': tgt_pad_mask
        }


def create_seq2seq_dataloader(src_sequences: List[List[int]], 
                               tgt_sequences: List[List[int]],
                               batch_size: int,
                               start_idx: int = 0,
                               end_idx: int = 1,
                               pad_idx: int = 3,
                               shuffle: bool = True,
                               num_workers: int = 0) -> DataLoader:

    dataset = Seq2SeqDataset(src_sequences, tgt_sequences, start_idx, end_idx)
    collator = Seq2SeqCollator(pad_idx)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )