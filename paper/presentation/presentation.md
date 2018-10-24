---
title: "IIIDYT at IEST 2018: Implicit Emotion Classification with Deep Contextualized Word Representations"
author: 
- Jorge A. Balazs, Edison Marrese-Taylor, Yutaka Matsuo
date: October 31, 2018

controls: "false"
transition: "none"
slideNumber: "true"
center: "false"
overview: "true"
width: |
    `"100%"`{=html}
height: |
    `"100%"`{=html}
margin: 0
minScale: 1
maxScale: 1

theme: white

fragments: "true"

link-citations: "true"
---

# Introduction {.center}

# Proposed Approach {.center}

# Preprocessing

![](../images/preprocessing_substitutions.png){width=80% height=80% .plain}

::: notes

- This was made mostly for sanity
- The replacements were chosen arbitrarily
- One reviewer asked what happened with shorter replacements; we found out that
  results did not change significantly

:::


# Architecture

![](../images/iest_architecture.png){width=80% height=80% .plain}

# Implementation Details and Hyperparameters

+----------------------+-------------------------------------------------+
| **ELMo Layer**       | Official implementation with default parameters |
+----------------------+-------------------------------------------------+
| **Dimensionalities** | ELMo output = $1024$                            |
|                      |                                                 |
|                      | BiLSTM output = $2048$ for each direction       |
|                      |                                                 |
|                      | Sentence vector representation = $4096$         |
|                      |                                                 |
|                      | Fully-connected (FC) layer input = $4096$        |
|                      |                                                 |
|                      | FC layer hidden = $512$                         |
|                      |                                                 |
|                      | FC layer output = $6$                           |
+----------------------+-------------------------------------------------+

# Implementation Details and Hyperparameters

+--------------------+------------------------------------------------------+
| **Loss Function**  | Cross-Entropy                                        |
+--------------------+------------------------------------------------------+
| **Optimizer**      | Default Adam                                         |
|                    | ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) |
+--------------------+------------------------------------------------------+
| **Learning Rate**  | Slanted triangular schedule ($cut\_frac=0.1,$ <br>   |
|                    | $ratio=32,\,\eta_{max}=10^{-3},\,T=23,970$)          |
|                    |                                                      |
|                    |                                                      |
+--------------------+------------------------------------------------------+
| **Regularization** | Dropout ($0.5$ after Elmo Layer and FC hidden;<br>   |
|                    | $0.1$ after max-pooling layer)                       |
+--------------------+------------------------------------------------------+


# Ensembles




<div class="flex-container" style="padding-top:5%;">

<div>
![](../images/best_ensembles.png){width=100% height=100% .plain}
</div>

<div class="fragment" style="padding-right:5%;text-align:justify;">
We tried $\sum_{k=1}^{9}{\binom{9}{k}}=511$ combinations of 9 trained models
initialized with different random seeds.

Similar to @bonab2016theoretical, we found out that ensembling 6 models
yielded the best results.
</div>

</div>


# Experiments and Analyses {.center}

# Ablation Study

<div class="flex-container">

<div>
![](../images/ablation_table.png){width=80% height=80% .plain}
</div>
<div style="padding-right:2%;text-align:justify;">
>- ELMo provided the biggest boost in performance.
>- Emoji also helped ([analysis](#effect-of-emoji-and-hashtags)).
>- Concat pooling [@howard2018universal], did not help.
>- Different BiLSTM sizes did not improve results.
>- POS tag Embeddings of dimension 50 slightly helped.
>- SGD optimizer with simpler LR schedule [@conneau2017supervised], did not help.
</div>

</div>


<!-- # Ablation Study

<div class="columns" style="display:table;">
<div class="column" style="width:50%;display:table-cell;vertical-align:middle;">
![Dropout](../images/dropout_table.png "Dropout"){width=80% height=80% .plain}
</div>
<div class="column" style="width:40%;display:table-cell;vertical-align:middle;text-align:justify;padding=2%;">

<div class="fragment" style="padding-right:3%;text-align:justify;">

Dropout concentrated around high values for word-level
representations and low values for sentence-level representations yielded better
results.

</div>

</div>
</div>
 -->

# Ablation Study


<div class="flex-container" style="padding-top:5%">

<div>
![Dropout](../images/dropout_table.png "Dropout"){width=100% height=100% .plain}
</div>
<div>

<div class="fragment" style="padding-right:3%;text-align:justify;">

Dropout concentrated around high values for word-level
representations and low values for sentence-level representations yielded better
results.

</div>

</div>
</div>

<!-- # Error Analysis

![Confusion Matrix](../images/confusion_matrix.png "Confusion Matrix"){width=50% height=50% .plain}
 -->
# Error Analysis

:::::: columns

<div class="column" style="width:50%;">
![Confusion Matrix](../images/confusion_matrix.png "Confusion Matrix"){width=70% height=70% .plain}
![Classification Report](../images/classification_report.png "Classification Report"){width=50% height=50% .plain}
</div>
<div class="column" style="padding-top:5%;text-align:justify;width:40%;">
>- `anger` was the hardest class to predict
>- `joy` was the easiest one 
   <div class="fragment">(probably due to an annotation artifact)</div>
</div>

::::::

# Error Analysis

<div class="flex-container">

<div>
![PCA projection of test sentence representations](../images/pca.png "PCA projection of test sentence representations"){width=80% height=80% .plain}
</div>
<div class="fragment" style="padding-right:5%;text-align:justify;">
Separate `joy` cluster corresponds to those sentences containing the
"un`[#TRIGGERWORD#]`" pattern.
</div>

</div>




# Effect of the Amount of Training Data

![](../images/acc_vs_tdp_variation.png){width=60% height=60% .plain}

# Effect of Emoji and Hashtags

<div class="flex-container" style="padding-top:5%;flex-direction:column">
<div>
![](../images/emoji_hashtag_performance.png){width=80% height=80% .plain}
</div>
<div>
<p class="fragment" style="padding:0 5%;text-align:justify;">
Number of examples with and without emoji and hashtags. Numbers between
parentheses correspond to the percentage of examples classified correctly.
</p>

<p class="fragment" style="padding:0 5%;text-align:justify;">
Tweets and hashtags, to a lesser extent, seem to improve performance.
</p>
</div>
</div>

::: notes

Overall effect of hashtags and emoji on classification performance.

Tweets containing emoji seem to be easier for the model to classify than those
without.

Hashtags also have a positive effect on classification performance, however it
is less significant.

This implies that emoji, and hashtags in a smaller degree, provide tweets with a
context richer in sentiment information, allowing the model to better guess the
emotion of the `trigger-word`.

:::

# Effect of Emoji and Hashtags

<div class="flex-container" style="padding-top:5%;">
<div style="flex:6">
![😷💕😍❤️😡😢😭😒😩😂😅😕](../images/fine_grained_performance.png){width=80% height=80% .plain}


</div>

<div style="flex:4;padding-right:3%;text-align:justify;">
>- `rage` 😡, `mask` 😷, and `cry` 😢, were the most informative emoji.
>- Counterintuitively, `sob` 😭 was less informative than 😢 despite
   representing a stronger emotion.
>- Removing `sweat_smile` 😅 and `confused` 😕 improved results.
</div>

</div>


# Conclusions and Future Work {.center}

# References
<div id="refs">
<!-- pandoc-citeproc will insert bibliography here -->
</div>


<!-- # In the morning

- Eat eggs
- Drink coffee

# In the evening

- Eat spaghetti
- Drink wine

# Fragments test

<p class="fragment grow">grow</p>
<p class="fragment shrink">shrink</p>
<p class="fragment fade-out">fade-out</p>
<p class="fragment fade-up">fade-up (also down, left and right!)</p>
<p class="fragment fade-in-then-out">fades in, then out when we move to the next step</p>
<p class="fragment fade-in-then-semi-out">fades in, then obfuscate when we move to the next step</p>
<p class="fragment highlight-current-blue">blue only once</p>
<p class="fragment highlight-red">highlight-red</p>
<p class="fragment highlight-green">highlight-green</p>
<p class="fragment highlight-blue">highlight-blue</p>

# Do columns work?

<div class="twocolumn">
<div>
- These
- Are
- Awesome super long elements to the left
</div>
<div>
- You can place two graphs on a slide
- Or two columns of text
- These are all created with div elements
</div>
</div>
Then what about a
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At
vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren,
no sea takimata sanctus est Lorem ipsum dolor sit amet.

# Conclusion

- And the answer is...
- $f(x)=\sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(x-a)^n$
 -->

<!-- Another way of creating two column slides -->
<!-- # Implementation Details and Hyperparameters

<div class="columns">
<div class="column" style="text-align:justify;width:20%;">

**ELMo Layer**

Optimizer

<br>
<br>
Learning Rate

</div>
<div class="column" style="text-align:justify;width:55%;">

Official implementation with default parameters

- Lorem ipsum dolor sit amet, 
- consetetur sadipscing elitr, sed diam nonumy

- Lorem ipsum dolor sit amet, 
- consetetur sadipscing elitr, sed diam nonumy

</div>
</div>

For citing: [@luong2016achieving]
 -->
