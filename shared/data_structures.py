# MaxPriorityMap implementation

class MaxPriorityMap:
    def __init__(self, heap_key, map_key):
        self._heap_key = heap_key
        self._map_key = map_key
        self._heap = []
        self._map = {}
    
    def get_max(self):
        if len(self._heap) == 0:
            return ValueError("The Heap is empty!")
        return self._heap[0]
    
    def push(self, item):
        self._heap.append(item)
        map_key = self._map_key(item)
        if map_key in self._map:
            raise KeyError("Item with the same map key already exists!")
        self._map[self._map_key(item)] = len(self._heap) - 1
        self._heapify_up(len(self._heap) - 1)

    def pop(self):
        if len(self._heap) == 0:
            return ValueError("The Heap is empty!")
        self._swap(0, len(self._heap) - 1)
        item = self._heap.pop()
        map_key = self._map_key(item)
        self._map.pop(map_key)
        self._heapify_down(0)
        return item
    
    def contains(self, map_key):
        return map_key in self._map
    
    def len(self):
        if len(self._heap) != len(self._map):
            raise ValueError("Invalid state: heap and map sizes do not match!")
        return len(self._heap)
    
    def delete_map_by_key(self, map_key):
        if map_key not in self._map:
            raise KeyError("Item with the given map key does not exist!")
        index = self._map[map_key]
        item = self._heap[index]
        self._swap(index, len(self._heap) - 1)
        self._heap.pop()
        self._map.pop(map_key)
        if index < len(self._heap):
            self._heapify_down(index)
            self._heapify_up(index)
        return item
    
    def _heapify_up(self, index):
        while index > 0 and self._heap_key(self._heap[index]) > self._heap_key(self._heap[self._parent(index)]):
            self._swap(index, self._parent(index))
            index = self._parent(index)
    
    def _heapify_down(self, index):
        largest = index
        left = self._left_child(index)
        right = self._right_child(index)

        if left < len(self._heap) and self._heap_key(self._heap[left]) > self._heap_key(self._heap[largest]):
            largest = left
        
        if right < len(self._heap) and self._heap_key(self._heap[right]) > self._heap_key(self._heap[largest]):
            largest = right
        
        if largest != index:
            self._swap(index, largest)
            self._heapify_down(largest)
    
    def _parent(self, index):
        return (index - 1) // 2
    
    def _left_child(self, index):
        return 2 * index + 1
    
    def _right_child(self, index):
        return 2 * index + 2
    
    def _swap(self, i, j):
        self._map[self._map_key(self._heap[i])] = j
        self._map[self._map_key(self._heap[j])] = i
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

# Linked_Array implementation

class Node:
    def __init__(self, value, previous, next, index):
        self.value = value
        self.previous = previous
        self.next = next
        self.index = index

class Linked_Array:
    def __init__(self, items):
        self._array = [None for _ in range(len(items))]
        previous = None
        for i in range(len(items)):
            node = Node(items[i], previous, None, i)
            if previous is not None:
                previous.next = node
            self._array[i] = node
            previous = node

    def get_by_index(self, index):
        if self._array[index] is None:
            raise ValueError("No Item By Index!")
        return self._array[index].value
    
    def get_previous_index(self, index):
        if self._array[index] is None:
            raise ValueError("No Item By Index!")
        if self._array[index].previous is None:
            return None
        return self._array[index].previous.index
    
    def get_next_index(self, index):
        if self._array[index] is None:
            raise ValueError("No Item By Index!")
        if self._array[index].next is None:
            return None
        return self._array[index].next.index
    
    def get_second_next_index(self, index):
        if self._array[index] is None:
            raise ValueError("No Item By Index!")
        if self._array[index].next is None or self._array[index].next.next is None:
            return None
        return self._array[index].next.next.index
    
    def len(self):
        return len(self._array)
    
    def replace_pair(self, index, new_item):
        if index > len(self._array) - 2 or self._array[index] is None or self._array[index].next is None:
            raise ValueError("Invalid index!")
        self._array[index].value = new_item
        self._array[self._array[index].next.index] = None
        self._array[index].next = self._array[index].next.next
        if self._array[index].next is not None:
            self._array[index].next.previous = self._array[index]
        