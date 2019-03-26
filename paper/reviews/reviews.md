Title: "Reviews for "IIIDYT at IEST 2018: Implicit Emotion Classification With
Deep Contextualized Word Representations"

Author: "Jorge Balazs, Edison Marrese-Taylor and Yutaka Matsuo"


# REVIEWER 1

## Reviewer's Scores
                   Appropriateness (1-5): 5
                           Clarity (1-5): 5
      Originality / Innovativeness (1-5): 3
           Soundness / Correctness (1-5): 4
             Meaningful Comparison (1-5): 5
                      Thoroughness (1-5): 5
        Impact of Ideas or Results (1-5): 4
                    Recommendation (1-5): 4
               Reviewer Confidence (1-5): 4

## Detailed Comments

This paper describes a BLSTM model with the recent EMLo embedding layer.  They
achieved the best result by ensembling 6 models with different random seeds.  It
is very nice that they also show the effect of emojis and hashtags. Especially
they give the impact of different emojis.

The paper is well structured and written. It demonstrates all the results
thoroughly. They tried to explain the implications of all the result.

However, it would be better if this paper discussed more about the results or
show some error analysis.


# REVIEWER 2

## Reviewer's Scores
                   Appropriateness (1-5): 5
                           Clarity (1-5): 4
      Originality / Innovativeness (1-5): 3
           Soundness / Correctness (1-5): 5
             Meaningful Comparison (1-5): 3
                      Thoroughness (1-5): 4
        Impact of Ideas or Results (1-5): 4
                    Recommendation (1-5): 4
               Reviewer Confidence (1-5): 5

## Detailed Comments

1) Appropriateness: This paper represents a very solid work done by the team for
the purpose of the IEST. 

2) Clarity: The paper is very clear and to the point. For those familiar with
   the subject it is very easy to read, which is important and at the same time
   very difficult to achieve. The structure of the paper and the analysis almost at
   all sections is really good. Also the level of English is high, very
   well-written paper. The clarity of the paper is enhanced by the figures and the
   tables with the results, especially figure 1 with the team's proposed
   architecture and the "heatmap" figure 3 with the dropout. 

   Although the idea for figure 2 with the correlation between the number of
   ensembled models and f1-score is interesting, I do not particularly believe it
   is prudent to have an even number of models for the final ensembling. How
   exactly is classification made when there is not a single label proposed by
   all-even-models? (For example 3 models classify a tweet as "sad" and 3 as
   "fear"...) It is true that this can also happen with a big number of odd models,
   e.g 7 or 9 or 11, !  but a clarification is a good idea at this section. 
 
   Moreover,  the implementation details (2.3) and the ablation study (3.1) provide
   useful technical details for those who train similar models. As far as the
   section "effect of the amount of training data" (3.3) is concerned, I believe it
   concluded to something rather trivial (more data provide best results) and
   figure 4 is somewhat difficult to understand. ("proportion of training data"? is
   it not evident that with 100% of training data the results are better than with
   less?) If the team intended to show that with less data they achieve competitive
   results, then it is not clear at all. 

3) Originality / Innovativeness:  It is clear that the authors are aware of the
   state-of-the-art models and how to use them. Unfortunately, the techniques
   proposed are not quite innovative. Their proposed architecture is, more or less,
   a simple biLSTM network with a pretrained ELMo layer. Although it is interesting
   using ELMo for emotion recognition, it is not considered an original/innovative
   idea or an important contribution to the bibliography. The team did not
   experiment with the ELMo embeddings, rather it seems they used them as a "black
   box". 

4) Soundness / Correctness: The work is sound, the results impressive and
   convincing, and the details of the model well-presented.

5) Meaningful Comparison: Although the presentation of the team's work is very
   satisfactory, they do not present almost any related work in the emotion
   recognition domain or NLP state-of-the-art models on similar tasks. It would be
   useful to extend their Introduction and provide information of existing
   approaches, baselines and compare their competitive results to highlight their
   contribution.

6) Thoroughness: The paper is sound and includes a lot of information. It would
   be preferable to include insight on other transfer learning techniques, except
   ELMo, or other approaches for tackling an emotion recognition task.

7) Impact of Ideas or Results: It is very promising that the use of a single ELMo
   layer provides such competitive results on the classification task. Surely this
   approach is inspirational and in the future it could be used by many to face
   many NLP tasks.

8) Recommendation: This paper is interesting and the results are truly
   impressive. There are issues indeed, but, if the suggestions presented will be
   taken into account, the resulting work will be very good.

# REVIEWER 3

## Reviewer's Scores

                   Appropriateness (1-5): 5
                           Clarity (1-5): 5
      Originality / Innovativeness (1-5): 4
           Soundness / Correctness (1-5): 5
             Meaningful Comparison (1-5): 3
                      Thoroughness (1-5): 5
        Impact of Ideas or Results (1-5): 4
                    Recommendation (1-5): 5
               Reviewer Confidence (1-5): 4

## Detailed Comments 

This paper describes a BiLSTM-based deep learning system for implicit emotion
classification. The main novelty is its use of pre-trained EMLo embeddings for
encoding words and the way of ensembling 6 different models to obtain the best
results.

The paper is well-written and enough details are included for the reader to
replicate the work, including the preprocessing used, network architecture and
hyperparameters, etc.

In particular, the authors included details analyses and ablation study to
provide interesting insights that are useful for the task and can be applied to
other tasks in general. E.g. contribution of ELMo layer, tweets with emoji can
be easier to classify, some emojis contribute more to improving accuracy.

There is a small part that can be further clarified: It is mentioned in Section
2.1 that specific tokens are substituted, e.g. "[#TRIGGERWORD#]" to
"__TRIGGERWORD__". Since the model is purely character-based, will there be any
significant difference if a shorter replacement string is used, e.g. "__T__"?
Another related question: if special tokens like "[#TRIGGERWORD#]" is totally
removed from the sentence (i.e. location of trigger word is not provided), what
is the effect on its performances?
