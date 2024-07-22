# homework 2
def count_char_frequency(word):
    frequency_dict = {}
    for char in word:
        char = char.lower()  # Chuyển sang chữ thường
        if char in frequency_dict:
            frequency_dict[char] += 1
        else:
            frequency_dict[char] = 1
    return frequency_dict


word = (input('enter your word:'))
print(count_char_frequency(word))
