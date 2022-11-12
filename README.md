# FAQ-matching

For solving this FAQ problem statement, I have to consider a few preliminary steps before moving forward with the models. First of all, I have to use regular expression techniques for cleaning the datasets. After that, I need to use different approaches so that I can make it acceptable to others.

I have a frequently asked questions list available in CSV format in the following file: It discusses Albert Einstein and includes a lot of questions and answers about him. In the datasets, it has both of the following features:


``` python

        1. Question
        2. Answer

```

There were two different datasets, one for training and another for testing or validating the model's performance.


<h4>I have taken two parts from transformer: </h4>

``` python
        1. BertForQuestionAnswering
        2. BertTokenizer
```

NLP's Transformer is a novel design that seeks to handle problems sequence-to-sequence while readily addressing long-distance dependencies. Because of this, the use of two transformer modules is required in order to take use of the architecture. For the purpose of determining the input and output representations, we do not make use of sequence-aligned RNNs or convolutions. Instead, we rely only on paying attention to ourselves.


``` python
        1. BertForQuestionAnswering
```

I have developed a new model for the representation of languages called BERT, which stands for Bidirectional Encoder Representations from Transformers. This model was developed specifically for this issue statement. BERT, in contrast to more contemporary models of language representation, is intended to pre-train deep bidirectional representations from unlabeled text. This is accomplished by simultaneously conditioning on both left and right context at all levels of the model. Therefore, the pre-trained BERT model can be fine-tuned with just one more output layer to make state-of-the-art models for a wide range of tasks, such as answering questions and making inferences about language, without having to make significant changes to the architecture for each task. This is possible because the pre-trained BERT model has already been trained.



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
