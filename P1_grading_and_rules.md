# Grading and Rules for Project 1 (Overdue)

## Rules for Choosing Data Sets

*   The dataset must have at least 1000 data samples.
*   If features are tabular, it must have at least 100 features (this does not apply to image, text, audio).
*   Must not be a popular dataset (e.g., MNIST, CIFAR, Iris, Movie Reviews, etc.).
*   Must have labels for interpretability / comparison with supervised method.
*   At most 1 student can choose a data set (history of previous evaluation is kept).
*   At most 1 student can choose a task (e.g., sentiment analysis, facial expression recognition, etc.).

## Rules for Choosing Methods

*   At most 4 students can choose the same model combination (history of previous evaluation is NOT kept).
*   At most 6 students can choose a certain model (history of previous evaluation is NOT kept).

## Grading Breakdown

Your grade will be based on the following points:

a) **0.5** - Granted  
b) **0.3** - Tests with 2 different supervised models (0.15 points per model), demonstrated by code  
c) **0.3** - Attempts with 2 different types of features (0.15 per representation), demonstrated by code (variations on a certain representation, e.g., bag-of-words with and without stopwords, are **NOT** sufficient)  
d) **0.2** - Tuning of parameters (including distance metrics where applicable), grid search, graphs showing the variation of performance depending on the values of the hyperparameters  
e) **0.2** – Confusion matrix for classification tasks / ordinal evaluation measure for regression tasks  
f) **0.1** - Comparison with method from literature  
g) **0.2** - Completeness of the documentation (at least 2 pages, excluding graphs, figures, tables)  
h) **0.2** - Correctness of approach (train, validation and test separation, i.e., hyperparameters are not set on the test)  
i) **0.2** - Compensation may be awarded if there are other additional aspects (testing with the third method, testing in transfer learning scenarios, etc.), if the sum from a) to j) is less than 2  

**Note:** The scores at **d), e), f), g), h)** are divided by 2 in case of implementing a single supervised method.

## Submission Instructions

Please submit the projects to: **practical.ml.fmi@gmail.com**, using the following formatting instructions:

*   Convert `*.ipynb` files to `*.py` (failure to comply will result in disqualification).
*   The `*.py` files must be placed in the same folder named `P1_{family_name}_{first_name}_{group}`. Subfolders with code are not allowed.
*   The report (in PDF format) must be placed in a subfolder named `P1_{family_name}_{first_name}_{group}_doc`.
*   Submitting the complete code is mandatory.
*   The folder containing the `*.py` files and the documentation must be archived in a single `*.zip` file and attached to the e-mail as a file (not link).
*   Do not attach data, only code!
*   Following the attached technical reporting guidelines is mandatory.

**Failure to comply with any of the above points will result in disqualification.**