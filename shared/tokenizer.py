from data_structures import Linked_Array
from data_structures import MaxPriorityMap

# To_Tokens_Converter implementation

class TokenNode:
    def __init__(self, basic_char: str, token: int, parent, children):
        self.parent = parent 
        self.basic_char = basic_char
        self.token = token
        self.children = children

class To_Tokens_Converter:
    def __init__(self, token_map: dict, chars_map: dict):
        self._token_map = token_map
        self._chars_map = chars_map
        self._unk_key = '<UNK>'
        self._init_token_tree()
    
    def to_tokens(self, string):
        tokens = []
        char_index = 0
        while char_index < len(string):
            if string[char_index] not in self._chars_map:
                tokens.append(self._chars_map[self._unk_key])
                char_index += 1
                continue
            current_nodes = self._token_roots
            last_node_with_token = None
            last_index_with_token = char_index
            while char_index < len(string):
                if string[char_index] in current_nodes:
                    node = current_nodes[string[char_index]]
                    if node.token is not None:
                        last_node_with_token = node
                        last_index_with_token = char_index
                    current_nodes = node.children
                    char_index += 1
                else:
                    break
            tokens.append(last_node_with_token.token)
            char_index = last_index_with_token + 1
        return tokens
    
    def _init_token_tree(self):
        self._token_roots = {}
        for token in self._token_map:
            string = self._token_map[token]
            current_nodes = self._token_roots
            last_node = None
            for char in string:
                if char in current_nodes:
                    last_node = current_nodes[char]
                else:
                    last_node = TokenNode(char, None, last_node, {})
                    current_nodes[char] = last_node
                current_nodes = last_node.children
            if last_node is not None and last_node.token is not None:
                raise KeyError("The token is already taken")
            if last_node is not None:
                last_node.token = token

# Tokenizer Trainer Implementation

class StatsEntry:
    def __init__(self, pair, positions):
        self.pair = pair
        self.positions = positions

class TokenizerTrainer:
    def __init__(self, input_as_basic_tokens, min_token_occurrance, tokens_map, max_iterations):
        self._input_as_basic_tokens = input_as_basic_tokens
        self._min_token_occurrance = min_token_occurrance
        self._tokens_map = tokens_map
        self._max_iterations = max_iterations

        self._str_to_token_map = {}

    def train(self, next_token):
        from utils import progressBar

        self._positions = [Linked_Array(basic_tokens) for basic_tokens in self._input_as_basic_tokens]
        self._calc_initial_stats()

        if self._max_iterations is None:
            self._max_iterations = 20000

        print("Starting BPE training...")
        pb = progressBar()
        pb.start(self._max_iterations)
        
        for iter in range(self._max_iterations):
            merge_stat = self._stats.pop()
            if len(merge_stat.positions) < self._min_token_occurrance:
                print(f"\nMin token occurrence achieved at iteration {iter+1}. Stopping Byte-Pair encoding")
                break
            (curr_token, next_token) = self._get_current_and_next_token(next_token, merge_stat)
            for pos in list(merge_stat.positions):
                if pos not in merge_stat.positions:
                    continue
                input_idx = pos[0]
                token_idx = pos[1]
                self._update_left_token(input_idx, token_idx, merge_stat, curr_token)
                self._update_right_token(input_idx, token_idx, merge_stat, curr_token)
                self._positions[input_idx].replace_pair(token_idx, curr_token)
                merge_stat.positions.remove((input_idx, token_idx))
            pb.tick()
        
        pb.stop()
        print("Training done")
    
    def _get_current_and_next_token(self, next_token, merge_stat):
        token_str_val = self._tokens_map[merge_stat.pair[0]] + self._tokens_map[merge_stat.pair[1]]
        current_token = -1
        if token_str_val in self._str_to_token_map:
            current_token = self._str_to_token_map[token_str_val]
        else:
            current_token = next_token
            self._tokens_map[current_token] = token_str_val
            self._str_to_token_map[token_str_val] = current_token
            next_token += 1
        return (current_token, next_token)
    
    def _update_right_token(self, input_index, token_index, merge_stat, new_token):
        positions = self._positions[input_index]
        second_token_index = positions.get_next_index(token_index)
        right_token_index = positions.get_second_next_index(token_index)
        if right_token_index == None:
            return
        pair = (merge_stat.pair[1], positions.get_by_index(right_token_index))
        self._remove_position_from_pair(merge_stat, pair, input_index, second_token_index)
        new_pair = (new_token, pair[1])
        self._add_position_to_pair(new_pair, input_index, token_index)
    
    def _update_left_token(self, input_index, token_index, merge_stat, new_token):
        positions = self._positions[input_index]
        left_token_index = positions.get_previous_index(token_index)
        if left_token_index == None:
            return
        pair = (positions.get_by_index(left_token_index), merge_stat.pair[0])
        self._remove_position_from_pair(merge_stat, pair, input_index, left_token_index)
        new_pair = (pair[0], new_token)
        self._add_position_to_pair(new_pair, input_index, left_token_index)

    def _remove_position_from_pair(self, merge_stat, pair, input_index, token_index):
        if pair == merge_stat.pair:
            merge_stat.positions.remove((input_index, token_index))
        else:
            stat = self._stats.delete_map_by_key(pair)
            stat.positions.remove((input_index, token_index))
            if len(stat.positions) != 0:
                self._stats.push(stat)
    
    def _add_position_to_pair(self, pair, input_index, token_index):
        stat = self._stats.delete_map_by_key(pair) if self._stats.contains(pair) else StatsEntry(pair, set())
        stat.positions.add((input_index, token_index))
        self._stats.push(stat)
    
    def _calc_initial_stats(self):
        self._stats = MaxPriorityMap(
            heap_key = lambda item: len(item.positions),
            map_key = lambda item: item.pair)
        stats = {}
        for string_ind in range(len(self._input_as_basic_tokens)):
            basic_tokens = self._input_as_basic_tokens[string_ind]
            for char_ind in range(len(basic_tokens)-1):
                pair = (basic_tokens[char_ind], basic_tokens[char_ind+1])
                if pair not in stats:
                    stats[pair] = StatsEntry(pair, set())
                stats[pair].positions.add((string_ind, char_ind))
        for key in stats:
            self._stats.push(stats[key])

# Tokenizer Implementation

class Tokenizer:
    def __init__(self, min_token_occurrance):
        self._min_token_occurrance = min_token_occurrance
        self._chars_map = {}
        self._tokens_map = {}
        self._converter = None

    def to_tokens(self, strings):
        if self._converter is None:
            self._converter = To_Tokens_Converter(self._tokens_map, self._chars_map)
        return self._converter.to_tokens(strings)

    def from_tokens(self, tokens):
        return [''.join(self._tokens_map[token] for token in token_str) for token_str in tokens]
    
    def train(self, strings):
        self._map_chars(strings)
        input_as_tokens = self._to_basic_token_ids(strings)
        trainer = TokenizerTrainer(
            input_as_tokens,
            self._min_token_occurrance,
            self._tokens_map,
            max_iterations=None)
        trainer.train(len(self._chars_map))
        self._converter = None
        return self._tokens_map
    
    def _map_chars(self, strings):
        self._chars_map = {}
        self._tokens_map = {}
        index = 0
        for text in strings:
            for char in text:
                if char not in self._chars_map:
                    self._chars_map[char] = index
                    self._tokens_map[index] = char
                    index += 1
        self._chars_map['<UNK>'] = index
        self._tokens_map[index] = '<UNK>'
        index += 1
        self._chars_map["<TRANS>"] = index
        self._tokens_map[index] = "<TRANS>"
    
    def _to_basic_token_ids(self, strings):
        return [[self._chars_map[char] if char in self._chars_map else self._chars_map['<UNK>'] for char in string] for string in strings]