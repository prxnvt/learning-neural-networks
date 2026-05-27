import torch 
import torch.nn.functional as F

# -----------------------------------------------------------------------------

words = open('names.txt', 'r').read().splitlines()

# The bigram counts the frequencies of consecutive character pairs 
b = {}
for w in words:
    chars = ['.'] + list(w) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        bigram = (char1, char2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(char1, char2)

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

# Creating training set of bigrams (x, y) for all words
xs, ys = [], [] 
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1] ; ix2 = stoi[ch2]
        xs.append(ix1) ; ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# Randomly initialize 27 neurons' weights. Each neuron receives 27 inputs 
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator = g, requires_grad = True)

# -----------------------------------------------------------------------------

# # Gradient Descent: 
epsilon = 0.0001
prev_loss = 0.0
counter = 0
while counter < 1000:

    # Forward pass: 
    xenc = F.one_hot(xs, num_classes = 27).float() # input to the network: one-hot encoding 
    logits = xenc @ W # predict log-counts 
    counts = logits.exp() # counts, equivalent to N 
    probs = counts / counts.sum(1, keepdims = True) # probabilities for next character 
    loss = -probs[torch.arange(num), ys].log().mean()

    # Check and update stopping conditions
    loss_f = float(loss.item())
    if (abs(loss_f - prev_loss) < epsilon):
        print(f'Converged to within ε = {epsilon}')
        break
    counter += 1 ; print(counter, loss.item())
    prev_loss = loss_f

    # Backward pass: 
    W.grad = None # Set the gradients to 0
    loss.backward()
    
    # Update gradients: 
    W.data += -50 * W.grad 


# for k in range(100):
    
#     # Forward pass: 
#     xenc = F.one_hot(xs, num_classes = 27).float() # input to the network: one-hot encoding 
#     logits = xenc @ W # predict log-counts 
#     counts = logits.exp() # counts, equivalent to N 
#     probs = counts / counts.sum(1, keepdims = True) # probabilities for next character 
#     loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
#     print(k, loss.item())

#     # Backward pass: 
#     W.grad = None # Set the gradients to 0
#     loss.backward()
    
#     # Update gradients: 
#     W.data += -50 * W.grad 

# -----------------------------------------------------------------------------

# Making up names: 
for i in range(5):
    out = []
    ix = 0

    while True: 

        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = xenc @ W
        counts = logits.exp() 
        p = counts / counts.sum(1, keepdims = True)

        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))