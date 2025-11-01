# Working with Text Data

Text Input need to be prepared for the LLM
Spitting text into individual word and subword tokens, which can then be encoded as vector representation

Reminder on Tensor

![alt text](image-4.png)

Converting data into vector is referred as embedding.
Specific Neural Network Layer of PreTrained Neural Network Model can be used (_a model in a model?_)

Different data format requires different embeddings models.

Word embedding is the most common form of text embedding, but can be done also for sentences, paragraphs or whole documents. Sentence or Paragraph embeddings are popular for RAG (Retrieval Augmented Generation)

One of earliest and popular example : **Word2Vec** &rarr; trained neural network architecture

Main idea : words that appear in similar context have similar meanings. In a 2D representation they can be clustered together. But word embeddings can have from 1 to 1000 dimensions.

While we can use pre-trained models like Word2Vec, LLMs commonly use their own embeddings as part of the input layer and they are updated during training, such that the embeddings are optimized to the specific task and data.

Hard to represent high density embeddings

Small GPT-2 : 786 Dimensions
Largest GPT3: 12,288 Dimensions

## Tokenizing text

Naive Tokenization by splitting by white spaces and punctuation. But no lowercasing or uppercasing of names as it helps the LLM to distinguish betwen proper nouns and common nouns and understand structure.

On removing whitespaces :

- Pros : Saves on memory and computing power
- Cons : Some text can be sensitive to whitespace, such as Python code

Basic Tokenizer by splitting :
`['I', 'HAD' .... '--', 'me', 'to', 'hear', 'that', ',', 'in']`

Once we have tokens we need to map them to a token which is an integer representation such that it can be understand by the model. -> This is the vocabulary

![alt text](image-5.png)


**How to handle unkown words in the vocab?**

### Adding special context tokens

`<|unk|>`
`<|endoftext|>`

![](image-6.png)

`<|unk|>` represents the unknown words of the vocabulary (*but only at inference time? does embedding model training needs to fix this?*)

`<|endoftext|>` represents the end of a text, useful when LLM trained on corpus of texts, since they are being concatenated for training while being unrelated

Depending on the LLM, researches have added other tokens : 

`[BOS]`: Beginning of sequence -> Start of text
`[EOS]`: End of sequence -> End of text
`[PAD]`: Padding -> When training LLMs with batch sizes larger than one, with text smaller than other ones. Those are being padded with `[PAD]`


GPT Models only use `<|endoftext|>` token for simplicity. Furthermore the GPT Tokenizer does not use the `<|unk|>` unknown token as it's leveraging Byte Pair Encoding, which breaks down words into subword units.

## Byte Pair Endoding
Vocabulary : 50,257 
The BPE tokenizer is capable of tokenizing unknown words by using subwords token

![alt text](image-7.png)

## Data sampling using sliding window

Since LLMs are trained to predict next word in a sentence we need to prepare a sliding window dataloader

```
context_size = 4 # How many tokens in the input
```

![](image-8.png)

Tensor size :  text_size -1 * context_size ? 