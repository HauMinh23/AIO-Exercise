
# Homework 1
from collections import deque
num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]


def max_sliding_window(num_list, k):
    if not num_list or k == 0:
        return []

    result = []
    window = deque()

    for i, num in enumerate(num_list):
        # Remove elements from the front of the window that are out of range
        while window and window[0] <= i - k:
            window.popleft()

        # Remove elements from the back of the window that are smaller than the current number
        while window and num_list[window[-1]] < num:
            window.pop()

        window.append(i)

        # Add the maximum element in the window to the result
        if i >= k - 1:
            result.append(num_list[window[0]])

    return result


print(max_sliding_window(num_list, 3))
