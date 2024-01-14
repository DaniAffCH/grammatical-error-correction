from evaluation import best_matching_ground_truth

predicted = "This is a test sentence."
ground_truths = [
    "This is a test sentence.",
    "This sentence is a test.",
    "This is not a test sentence.",
    "Testing this sentence."
]

best_ground_truth, best_score = best_matching_ground_truth(predicted, ground_truths)
print(f"Best Matching Ground Truth: {best_ground_truth}\nScore: {best_score}")
