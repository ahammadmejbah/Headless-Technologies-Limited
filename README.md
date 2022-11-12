# FAQ-matching






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
