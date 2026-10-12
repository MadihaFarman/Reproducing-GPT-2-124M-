from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#-----------------------------
## task 1 : lets load the gpt2 124M into the class that we are going to develop here. 
## gpt2 is different from original attention is all you need paper architecture in a few ways. Num 1: reshuffling of layer norms and Num 2: addition of a new norm layer before final classifier. Num 3: Its decoder only, so the cross attention block is completely removed.

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   # the assert keyword in Python is a debugging aid that tests if a specific condition in your code evaluates to True. If the condition passes, the program continues executing normally; if it evaluates to False, Python immediately halts the program and raises an AssertionError
        # key, query , value projection for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd,3* config.n_embd)
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # it takes in 768 features and outputs 768 features (meaning the size doesn't change), it is a fully connected linear layer with a weight matrix of shape [768, 768].# When you pass your concatenated tensor through this layer:# Every single output feature from Head 1 gets multiplied and combined with features from Head 2, Head 3, and all the others.It acts as a linear mixer, blending the perspectives of all the different attention heads together into a single, unified representation before it leaves the attention block.
        # regularization .. Saving configuration variables as class attributes (self.n_head and self.n_embd) to be used in forward pass i.e (B,T,self.n_head,C // self.n_head)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))  # .view(1, 1, config.block_size, config.block_size):Reshapes the 2D matrix into a 4D tensor of shape [1, 1, 1024, 1024]. This extra dimension padding allows it to broadcast seamlessly across different batches and multiple attention heads during matrix operations.

    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)   # (B,T,3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)   # (B,T,n_embd/C)
        # we are making the number of heads i.e nh into a batch dimension,  so its a batch dim just like B, so that in the operations that follow, pytorch treats B and nh as batches and applies the operations in parallel in both the batch and the heads.
        # nh is "num of heads", hs is "head size" and C (number of channels) = nh*hs
        # .view() (A method that reshapes a tensor without copying its underlying memory.)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        att = (q @ k.transpose(-2,-1)* (1.0/math.sqrt(k.size(-1))))   
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))  # wherever the bias is 0 (meaning: "Block this position because it's in the future") 
        # : (Dimension 1): Keep all elements along the batch dimension (which is size 1).: (Dimension 2): Keep all elements along the attention head dimension (which is size 1).:T (Dimension 3): Take rows from index 0 up to the current sequence length $T$.:T (Dimension 4): Take columns from index 0 up to the current sequence length $T$.
        att = F.softmax(att,dim=-1)  # softmax normalizes attention so it always sums to 1.
        y = att @ v   # a way of doing weighted sum   (B, nh, T, hs) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)  # reassembles all head outputs side by side through .view(B,T,C) (the concatenation operation)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)    # the 4* comes from original paper where (512*4 = 2048)... n_embd is the incoming input sixe, 4*n_embd is the outgoing dim size and also represent num of nodes in neural layer
        self.gelu = nn.GELU(approximate='tanh')    # GELU is almost like relu, here the original gelu is not used, rather a tanh approximation function of gelu is used,the reason is that at the time of dev of gpt, the original gelu evalutaion function in tensor was very slow, but not now,but still most models use gelu approximate func.
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))   # here is change from original paper, layernorm is applied before attention and ffwd rather than after attention and ffwd as in paper. This allows for clean residual pathway and helps in optimzation  ...attention can be thought as reduce function where all the tokens communicate/echange information, its an aggregation function/reduce function
        x = x + self.mlp(self.ln_2(x))    # mlp =  ffwd/ffnn   ... mlp can be thought as it happens at every token individually, there is no info collected or exchanged between tokens, its a sort of map function, (think individually about info they gathered)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12     # blocks
    n_head: int = 12
    n_embd: int = 768 

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # the basic skeleton.architecture 
        self.transformer = nn.ModuleDict(dict(  # the main container = transformer ... playground.ipynb cell 2 output
            wte = nn.Embedding(config.vocab_size,config.n_embd),   # nn.Embedding is a glorified wrapper around tensor that allows you to access its element by indexing into rows
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h's are the encoder blocks
            ln_f = nn.LayerNorm(config.n_embd)  # additional final layer norm (transformer doesnt have it, GPT2 has added this extra)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias=False)   # final classifier--the lang model head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight   # this is present in original transformer paper but we missed.. note for myself : check in detail plz

    def forward(self, idx):
        # idx is of shape (B,T)
        B,T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T} as maximum token limit is {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape : T
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits


   # load GPT2 params from hugging face to our model and initalize GPT class with those parameters
    @classmethod
    def from_pretrained(cls,model_type):   # the form_pretrained is a constructor or class method in python that returs the GPt object if we give it the model_type 
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel   # Imports Hugging Face's pre-built GPT-2 model class locally so we can download their official weights.
        print("loading weights from pretrained gpt: %s"  % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),           # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),          # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),          # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),          # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257   # always use 50257 for GPT model checkpoint
        config_args['block_size'] = 1024    # always use 1024 for GPT model checkpoint
        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)  # Instantiates your custom-built GPT model from scratch using those configuration settings, creating an architecture filled with randomly initialized weights.
        sd = model.state_dict()  # pulls out a dictionary of all tensors (weights and biases) belonging to your custom model.
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask/buffer --- not a learnable parameter.

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)   # Downloads OpenAI's official pre-trained GPT-2 model and its weights from Hugging Face into your computer's memory.
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the params are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
         # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    # special treatment for the Conv1D weights we need to transpose
                    assert sd_hf[k].shape[::-1] == sd[k].shape   # Verifies that reversing the shape of the Hugging Face weight tensor matches the expected shape of your custom model's tensor.
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())   # Transposes the Hugging Face weight matrix using .t() to fix the Conv1D layout difference, and copies those values directly into your custom model's corresponding tensor using .copy_().
                else:
                    # vanilla copy over the other parameters
                    assert sd_hf[k].shape == sd[k].shape   
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])   # Catches all other parameters that do not require transposition (such as layer normalization weights/biases, embedding tables, and output heads).

        return model

# -----------------------------------------------------------------------------------

# model = GPT.from_pretrained('gpt2')
# print("didn't crash yay!!")

# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device: {device}")
## let's generate from the model
num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')  # importing pretrained weights
# initalizing model from scratch 
model = GPT(GPTConfig())  # the output generated on first time would be random....when we create GPT mdoel, all the layers get created with default constructors to construct the weights of these layers
model.eval()   # put the model to eval mode whenever using it. Some layers have diff behaviours in training vs eval (not in this case)
# model.to('cuda')   # moving all tensors to GPU, with this line we move the entire model, and all of its tensors to a separate computer which is gpu, that is connected to cpu and is a whole seprate computer
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # 8
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)   # replicate the 8 tokens to get 5 same rows 
x = tokens.to('cuda')   # x is the idx to forward to get our logits so that we know what comes as this 9th token in everyone of these 5 rows.


# generate !!!!  right now the sixe of x is (B,T)
# appending every generated new idx (token) during each loop operation to the sequence.
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():   # telling pytorch we are not gonna call backprop on it so dont save states
        logits = model(x)   # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:,-1,:]  # (B, vocab_size)
        # get the prob
        probs = F.softmax(logits,dim=-1)
        # do top-k sampling of 50 (like huggingface pipeline default)   : topk_probs: take probs and keep only topk probs
        # topk_probs here becomes (5,50), topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs
        ix = torch.multinomial(topk_probs, 1)   # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)