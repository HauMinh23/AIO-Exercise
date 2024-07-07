# Homework 4
class MyQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def enqueue(self, value):
        if self.is_full():
            print("Queue is full!")
            return
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty!")
            return None
        value = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return value

    def front(self):
        if self.is_empty():
            print("Queue is empty!")
            return None
        return self.queue[self.front]


# Example usage
queue1 = MyQueue(5)

# Enqueue elements
queue1.enqueue(1)
queue1.enqueue(2)

# Check if queue is full
print(queue1.is_full())  # Output: False

# Dequeue an element
print(queue1.dequeue())  # Output: 1

# Dequeue another element
print(queue1.dequeue())  # Output: 2

# Check if queue is empty
print(queue1.is_empty())  # Output: True
