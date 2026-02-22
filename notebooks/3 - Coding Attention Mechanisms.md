

## 3.1 Problem with modeling long sequences

Self Attention

Translating word by word -> not working

Encoder/Decoder

RNN : Recurrent Neural Network
- Type of Neural Network where outputs from previous steps are fed as inputs in the current step, well suited for sequential data.

Encoder : Input to hidden states
Decoder : Hidden states to Output

![alt text](image-11.png)

Issues and limitations: 
- The RNN can not access earlier hidden states, only the current one that has all the relevant information. Leads to loss of context, especially in long and complex sentences. (Wich we stil have nowadays? Contex Rotting?)



## 3.2 

Bahdanau attention 

Bahdanau attention modifies the encoder/decoder RNN such that the decoder can access selectively different parts of the input sequence.

![alt text](image-12.png)

This selection is done using Attention Weights 

Self Attention is a mechanism that allow each position in input to attend all positions in same sequence when computing representation of a sequence.


## 3.3

### Self-Attention
-> Self refers to the mechanism ability to compute attention weights by relating the different positions within a single input sequence. Learns relationships and dependencies between the elements of the single sequence

Introspection?


Example :

For input x2 

![alt text](image-13.png)

```
1 -> t
z2 = a21*x1 + a22*x2 + a23*x3 + ... + a2t * xt
```

In self-attention, our goal is to calculate context vectors z(i) for each element x(i) in the
input sequence. A context vector can be interpreted as an enriched embedding vector.


Dot Product 

![alt text](image-14.png)

![alt text](image-15.png)

![alt text](image-16.png)

![alt text](image-17.png)

dot product between the input sequence and the query.
Higher dot product means higher similiarity. The dot product quantifies how much two vectors are aligned.

Once the attention score is computed -> we can get the attention weights by normalizing the score  such that it sums up to 1. 

Common : softmax, 
![alt text](image-18.png)

![alt text](image-22.png)

Also ensure result is positive

![alt text](image-19.png)

Calculatin vector z(2) by multiplying the embedded input tokens x(i) with the attention weights and then summing the resulting vectors

### 3.3.2 : All Weights togethers 

![alt text](image-20.png)


Step 1 : Compute Attention Scores
-> Compute dot product of inputs against each others
Step 2 : Compute Attention Weights
-> Normalization
Step 3 : Compute context vectors
-> Weighted sum over the inputs


![alt text](image-21.png)

Attention Scores: 

Multiplication by transpose

`inputs @ inputs.T`

Steps :
```
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1) 
all_context_vecs = attn_weights @ inputs
all_context_vecs
```

When doing matrix multiplication, ouput of is of size rows of first column by number of columns of second matrix

## 3.4 Self Attention with trainable weights

Also called `scalde dot-product attention`

Trainable weights are key as they are crucial for the model to learn how to produce "good" context

Introducing trainable weights matrices 

```
Wq -> query
Wk -> key 
Wv -> value 

```

![alt text](image-23.png)