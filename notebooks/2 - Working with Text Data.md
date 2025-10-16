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

Once we have tokens we need to map them to a token which is an integer representation such that it can be understand by the model. This is the vocabulary

![alt text](image-5.png)