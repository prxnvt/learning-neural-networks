import torch 

# -----------------------------------------------------------------------------

words = open('names.txt', 'r').read().splitlines()

# -----------------------------------------------------------------------------

# The bigram counts the frequencies of consecutive character pairs 
b = {}
for w in words:
    chars = ['.'] + list(w) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        bigram = (char1, char2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(char1, char2)

# Sorting the pair:frequency tuples
# print(sorted(b.items(), key = lambda kv : -kv[1]))

# -----------------------------------------------------------------------------

# Initialize 28x28 array 
N = torch.zeros((27, 27), dtype = torch.int32)

# Create sorted list of characters a, ..., z, <S>, <E> corresponding to 1, ..., 27; and one for vice-versa 
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)} ; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    # sandwiching the word's characters between <S> and <E>
    chs = ['.'] + list(w) + ['.']

    # isolate, and index through each consecutive character pair 
    for ch1, ch2 in zip(chs, chs[1:]):
        # find the corresponding integers of these two characters
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        # append the value stored in the cell (ix1, ix2) by 1
        N[ix1, ix2] += 1

# -----------------------------------------------------------------------------

