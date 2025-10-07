## Understanding Large Language Models 
------------------

- LLMs utilize an architecture called the transformer
which allows them to pay selective attention to different parts of the input when
making predictions

- Since LLMs are capable of generating text, LLMs are also often referred to as a form of
generative artificial intelligence (AI), often abbreviated as generative AI or GenAI.

- Instead of manually writing rules to identify spam emails, a machine learning algorithm is fed
examples of emails labeled as spam and legitimate emails. By minimizing the error in its
predictions on a training dataset, the model then learns to recognize patterns and
characteristics indicative of spam, enabling it to classify new emails as either spam or
legitimate.
As illustra

- PreTraining vs Finetuning 
- PreTrained -> Foundational Model

- Research has shown that when it comes to modeling performance, custom-built LLMs
those tailored for specific tasks or domains can outperform general-purpose LLMs, such as
those provided by ChatGPT, which are designed for a wide array of applications. 

- The two most popular categories of finetuning LLMs include **instruction-finetuning** and
finetuning for **classification** tasks. 

    - In instruction-finetuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. 
    - In classification finetuning, the labeled dataset consists of texts and associated class labels, for example, emails associated with spam and non-spam labels.