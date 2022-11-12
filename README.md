<h1><code>Headless Technologies Limited</code></h1>


<h3><b> Problem Statement Description: </b></h3>

In today's environment, the way information is accessible may be greatly influenced by the use of frequently asked questions (FAQ) systems. Within the field of computer science, the field of FQA may be located at the point where information retrieval and natural language processing meet. The creation of intelligent computer systems that are able to react in natural language to questions posed by users is the focus of this area of study. FQA systems that are able to retrieve desired information accurately and quickly without having to skim through the entire corpus can serve as valuable assets in industry, academia, and our own personal lives. This can result in improved search engine results and more personalized conversational chatbots, for example. Law companies and human resources departments are two examples of real-world settings in which FQA systems have the potential to be useful. In the first scenario, specific information may be required (for citation in a current case) from the records of thousands of previous cases; nonetheless, adopting a FQA system may save a significant amount of time and labor when compared to other options. In the latter scenario, an employee may need to look up particular laws, such as those pertaining to vacations; employing a FQA system might assist them in this endeavor by providing the appropriate answers to queries of the type "How many job openings are available each year?"


<h3><b> Solution Procedure: </b></h3>
        
For solving this FAQ problem statement, I have to consider a few preliminary steps before moving forward with the models. First of all, I have to use regular expression techniques for cleaning the datasets. After that, I need to use different approaches so that I can make it acceptable to others.

I have a frequently asked questions list available in CSV format in the following file: It discusses Albert Einstein and includes a lot of questions and answers about him. In the datasets, it has both of the following features:


``` python

1. Question
2. Answer

```

There were two different datasets, one for training and another for testing or validating the model's performance.


<h3><code> For solving this problem statement I have used bellow packages from python:</code></h3>

``` python

1. Fast Sentence Embeddings --> fse
2. Gensim --> gensim
3. Natural Language Toolkit --> nltk
4. Scikit-learn --> sklearn
5. Regular Expressions --> re
6. Uni Code Data --> unicodedata
7. Matplotlib 
8. Seaborn 
9. Pytroch --> torch
10. Transformers

```

<h4><code>Fast Sentence Embeddings" (FSE) :</code></h4>

``` python

import fse

```

There are several different embedding approaches that we may employ to successfully incorporate the meta data of our query. In my experience, I have made use of "Fast Sentence Embeddings," which is a Python package that may be thought of as an extension of Gensim. The purpose of this library is to make it as simple as possible to locate sentence vectors for big sets of sentences or texts by using vectors.

<h4><code>I have taken two parts from transformer:</code></h4>

``` python
1. BertTokenizer
2. BertForQuestionAnswering
        
```

NLP's Transformer is a novel design that seeks to handle problems sequence-to-sequence while readily addressing long-distance dependencies. Because of this, the use of two transformer modules is required in order to take use of the architecture. For the purpose of determining the input and output representations, we do not make use of sequence-aligned RNNs or convolutions. Instead, we rely only on paying attention to ourselves.

</br>
</br>

``` python
                                                 1. BertTokenizer
```


The BertTokenizer was yet another job that was finished for this problem statement. A tool that is referred to as a "word fragment tokenizer" is used by BERT. The process works by disassembling words into either their whole forms (for example, one word becomes one token) or into word parts, with the possibility that a single word may be broken up into several tokens. One scenario in which this may be helpful is one in which a word can be written in more than one way.

``` python

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

```
</br>
</br>

``` python
                                                 2. BertForQuestionAnswering
```

I have developed a new model for the representation of querstion and answers called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT, in contrast to more contemporary models of language representation, is intended to pre-train deep bidirectional representations from unlabeled text. This is accomplished by simultaneously conditioning on both left and right context at all levels of the model. Therefore, the pre-trained BERT model can be fine-tuned with just one more output layer to make state-of-the-art models for a wide range of tasks, such as answering questions and making inferences about language, without having to make significant changes to the architecture for each task. This is possible because the pre-trained BERT model has already been trained.

``` python
model_bert = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

```
</br>
</br>

<h2>I used cosine similarity to compare the vectors here.</h2>
<h3><code> Cosine Similarity </code></h3>

The degree of resemblance between two vectors in an inner product space may be measured using the cosine similarity. It checks if two vectors are heading nearly in the same direction by measuring the cosine of the angle formed by the two vectors and comparing the results. In text analysis, it is often used as a measurement tool for document similarity.


``` python

def cosine_similarity(X, Y=None, dense_output=True):

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

    return K

```
