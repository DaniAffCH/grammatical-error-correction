# gec_evaluation.py

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')

def best_matching_ground_truth(predicted_sentence, ground_truths):

    predicted_tokens = word_tokenize(predicted_sentence)
    scores = []

    for gt in ground_truths:
        reference_tokens = word_tokenize(gt)
        score = sentence_bleu([reference_tokens], predicted_tokens)
        scores.append(score)

    max_score_index = scores.index(max(scores))
    return ground_truths[max_score_index], scores[max_score_index]
