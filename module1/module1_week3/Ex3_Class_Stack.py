# Homework 3
class MyStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = [None] * capacity
        self.top = -1

    def is_empty(self):
        return self.top == -1

    def is_full(self):
        return self.top == self.capacity - 1

    def pop(self):
        if self.is_empty():
            print("Stack is empty!")
            return None
        value = self.stack[self.top]
        self.stack[self.top] = None
        self.top -= 1
        return value

    def push(self, value):
        if self.is_full():
            print("Stack is full!")
            return
        self.top += 1
        self.stack[self.top] = value

    def top(self):
        if self.is_empty():
            print("Stack is empty!")
            return None
        return self.stack[self.top]
