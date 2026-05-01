---
title: "Bag of words - Under the hood"
date: 2019-05-31
last_updated: 2019-05-31
categories: [machine_learning]
tags:
  - nlp
  - hashing
  - preprocessing
---

Before feeding textual data into a machine learning model, we need to extract features from it, which are then used as inputs to the model. In this article, we will go over one of the most commonly used feature encoding techniques: ***Bag of words*** (BoW). Given some text, *bag of words* representation will return *vocabulary of words (or features)* extracted from the text and a *vector of values* associated with these words.





### Count Vectorization:

It's the most simple form of bag of words' representation, which returns *vocabulary of words* and the related *frequency vectors*. Let's try an example. For corpus with 3 documents:

***Document 1***:	The cat got chased by dog

***Document 2***:	The cat jumped on a tree

***Document 3***:	The dog kept barking beneath the tree



The BoW representation will have following features/vocabulary:

```
['the', 'cat', 'got', 'chased', 'by', 'dog', 'jumped', 'on', 'tree', 'kept', 'barking', 'beneath']
```

And for each document, there will be a frequency vector with same length as total features, showing the frequency for each feature in a document:

```
The cat got chased by dog	=>	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
The cat jumped on a tree	=>	[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
The dog kept barking beneath the tree	=>	[2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
```



It's almost always the case that ***stop words*** are removed from text before applying BoW transformation. *Stop words* are words like *a, the, on, and, by*, that have no meaningful value but will increase the feature dimension.  There are also other data cleaning techniques that are commonly applied to improve the model quality like removing punctuations, making all words small case, stemming/lemmatization, etc





### N-Grams

Lot of entities are there which are named using multiple words e.g. bus stop, high school, happy birthday, Four seasons hotel, etc. If they are converted into single word features, they will lose their context meaning. We use n-gram to avoid this issue. ***n-grams*** can be thought of as a technique to group **n** number of words and build vocabulary using these grouped words. In case of *bigram* (or 2-gram), features of above corpus would look like:

```
['the cat', 'cat got', 'got chased', 'chased by', 'by dog', 'cat jumped', 'jumped on', 'on tree', 'the dog', 'dog kept', 'kept barking', 'barking beneath', 'beneath the', 'the tree']
```



The frequency vectors will now have frequency for these word groupings. It's also possible to build a vocabulary that contain with combination of both, single words and n-grams.





### TF-IDF

The problem with above approach is that words with highly frequency tend to hold more weight, even if they are not that relevant, for example, words like ***seen***, ***asked*** and ***went*** are pretty common in news articles and if you were to build news recommendation system where these words are given more weight, your system would recommend. Here the system would recommend articles with these words in them. This is not something you want. On the other side, model where more weight is given to low frequency words like ***soccer***, ***president*** and ***Jurassic***, articles containing these words would be recommended and users are lot more likely to open them. This can be achieved by using TF-IDF vectorization. Here

**TF** stands for *term frequency*. It's a frequency for a term in some document.

**IDF** stands for *Inverse Document Frequency*. It's a measure for how common the term is, across all the documents in corpus.


\\[
\large w_{i,j} =  tf_{i,j} * log (\frac{N}{df_{i}})
\\]

Here 

N = Total documents,

\\( \large w_{i,j} \\)  = Weight for \\( \large i_{th} \\) term in \\( \large j_{th} \\) document,

\\( \large tf_{i,j} \\) = Term frequency for \\( \large i_{th} \\) term in \\( \large j_{th} \\) document,

\\( \large df_{i} \\)   = Document frequency i.e. No. of documents with the \\( \large i_{th} \\) term








### Hashing Trick

In real world, Bag of words representation is highly prone to exploit. A popular example of this is how, in early days, spammers used to evade spam filters by using words that are not in vocabulary of classification model like ***PR1ZE, w0n, thousandz***, etc. To fix this issue, you would need to include these words in your model vocabulary which isn't as easy as it may sounds. Each time you include new words to vocabulary, the resulting feature vectors’ size will change and you would need to retrain the model to expect feature vectors with this new size. 

Moreover, there's an issue of *dimensionality curse*. As you add new features, model will keep getting harder to train, more demanding on resources and will start *over-fitting*. That's where the *feature hashing*, also called *hashing trick*, comes into play. 

***Hashing Trick uses hashing function, which maps any arbitrary size data to a fixed size data***. For example, applying **MD5** hash function on our first document will return:

```
MD5Hash('The cat got chased by dog')	=>	db022bfeaa78f30c9e419b75fc490adf
```



The cool thing about hashing function's that given the same input, it will always return same output. That's why it's also commonly used to authenticate binary files after downloading from remote source. We can use the hashing trick to map our feature vectors to **'Hash Function'** and use the *hashed vector* as input to model. This way, if any new *word* gets introduced to vocabulary, our input vector size will remain same.

Hashing trick makes it quite easy to scale our model to fit large datasets but as the dataset size increases, we will eventually encounter ***hash collision***. Hash collision occurs when hashing function returns the same output value for two different inputs. It's happens because you have more items to hash than your slots in hash table. To avoid this issue, we can set the output size to be a large number (*Sklearn uses 2**20 as default*).





### Final Thoughts

We have seen how Bag of words representation can be used to extract features for modeling. Although it's quite a popular technique but there's also a downside to it i.e. it ignores the semantics of document. To overcome this shortcoming, alternative techniques can be used like word2vec, LDA, etc.
