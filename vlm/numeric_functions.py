import os
import json
import pandas as pd
import difflib
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer, scoring
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider_scorer import CiderScorer


# Calculate ROUGE score
def calculate_rouge(reference_texts, candidate_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(candidate_text, ref) for ref in reference_texts]
    average_scores = {
        key: sum(score[key].fmeasure for score in scores) / len(scores)
        for key in scores[0]
    }
    return average_scores['rougeL']


def calculate_meteor(reference_texts, candidate_text):
    meteor_scorer = Meteor()
    score, _ = meteor_scorer.compute_score(reference_texts, [candidate_text])
    return score

def normalize_score(score, min_score=0, max_score=1):
    return (score - min_score) / (max_score - min_score)

