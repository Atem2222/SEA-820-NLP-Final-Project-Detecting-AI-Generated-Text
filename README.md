# Detecting AI-Generated Text

## SEA 820 Natural Language Processing Final Project

### Team Members

- Student 1: **Atem Ako Eyong Atem** — **113714224**
- Student 2: **Nahaeli Brunder** — **145268223**

---

## Project Overview

The rapid adoption of large language models has made it increasingly difficult to distinguish between human-written and AI-generated text.

The goal of this project is to build, evaluate, and compare two natural language processing approaches for classifying text as:

- `0` — Human-written
- `1` — AI-generated

The project compares:

1. **TF-IDF with Logistic Regression**
2. **Fine-tuned DistilBERT**

The evaluation includes accuracy, precision, recall, F1-score, confusion matrices, TF-IDF feature analysis, false-positive and false-negative analysis, shared and model-specific errors, text-length analysis, and ethical considerations tied to the observed results.

---

## Repository Structure

```text
SEA820-AI-Text-Detection/
│
├── SEA820_Final_Project_AI_Text_Detection_SUBMISSION_READY.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── final_model_comparison.csv
│   ├── distilbert_experiment_log.csv
│   └── final_model_predictions_and_errors.csv
│
├── report/
│   └── SEA820_Final_Report.pdf
│
└── presentation/
    └── SEA820_Final_Presentation.pdf
```

The full dataset is not included because of its size.

---

## Dataset

This project uses an **AI vs. Human Text** dataset from Kaggle.

### Original Dataset

Add the exact Kaggle dataset link here:

```text
https://www.kaggle.com/datasets/arjunverma2004/ai-vs-human-text-balanced-180k-records?resource=download
```

The original dataset contained hundreds of thousands of text samples. To make Transformer fine-tuning manageable, a balanced sample of **40,000 texts** was created:

- 20,000 human-written texts
- 20,000 AI-generated texts

The processed dataset used by the notebook is:

```text
ai_human_sample_40000.csv
```

Expected columns:

```text
text
label
```

Label definitions:

```text
0 = Human-written
1 = AI-generated
```

### Why the Dataset Is Not Included

The dataset is too large for normal GitHub storage and may also be subject to Kaggle licensing conditions.

To reproduce the project:

1. Download the original dataset from Kaggle.
2. Run the sampling code below.
3. Place `ai_human_sample_40000.csv` in the same folder as the notebook.

### Recreate the 40,000-Text Sample

```python
import pandas as pd

input_file = "AI_Human_balanced_dataset.csv"
output_file = "ai_human_sample_40000.csv"

df = pd.read_csv(input_file)

if "label" in df.columns:
    label_column = "label"
elif "generated" in df.columns:
    label_column = "generated"
else:
    raise ValueError(
        f"Label column not found. Available columns: {df.columns.tolist()}"
    )

if "text" not in df.columns:
    raise ValueError(
        f"Text column not found. Available columns: {df.columns.tolist()}"
    )

df = df[["text", label_column]].dropna()
df = df.drop_duplicates(subset="text")
df[label_column] = df[label_column].astype(int)

human = df[df[label_column] == 0].sample(
    n=20000,
    random_state=42
)

ai_generated = df[df[label_column] == 1].sample(
    n=20000,
    random_state=42
)

sampled_df = pd.concat(
    [human, ai_generated],
    ignore_index=True
)

sampled_df = sampled_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

sampled_df = sampled_df.rename(
    columns={label_column: "label"}
)

sampled_df.to_csv(output_file, index=False)

print(sampled_df.shape)
print(sampled_df["label"].value_counts())
```

Expected output:

```text
(40000, 2)

label
0    20000
1    20000
```

---

## Data Quality

The final sampled dataset contained:

- 40,000 total texts
- 20,000 human-written texts
- 20,000 AI-generated texts
- No missing values
- No duplicate texts

Human-written texts averaged approximately **422.1 words**, while AI-generated texts averaged approximately **344.1 words**.

This difference was treated as a possible dataset shortcut and examined further during error analysis.

---

## Data Split

A stratified 80/10/10 split was used:

- Training: 32,000 texts
- Validation: 4,000 texts
- Testing: 4,000 texts

The same split was used for both models.

A fixed random seed was used:

```python
SEED = 42
```

---

## Model 1: TF-IDF + Logistic Regression

The classic baseline uses:

- Lowercasing
- English stop-word removal
- Unigrams and bigrams
- Maximum vocabulary size of 50,000 features
- Minimum document frequency of 3
- Sublinear term frequency
- Logistic Regression with a maximum of 1,000 iterations

This model is fast, interpretable, and provides a strong baseline.

---

## Model 2: DistilBERT

The Transformer model uses:

- `distilbert-base-uncased`
- Hugging Face Transformers
- Hugging Face Datasets
- Trainer API
- Dynamic padding
- Maximum sequence length of 256 tokens
- Early stopping
- Validation F1-score for model selection

Two hyperparameter configurations were tested.

### Experiment A

- Learning rate: `2e-5`
- Training batch size: `8`
- Evaluation batch size: `16`
- Epochs: `2`

### Experiment B

- Learning rate: `3e-5`
- Training batch size: `16`
- Evaluation batch size: `32`
- Epochs: `2`

The best configuration was selected using validation F1-score only.

---

## Final Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 98.67% | 99.34% | 98.00% | 98.67% |
| DistilBERT | 99.30% | 99.20% | 99.40% | 99.30% |

DistilBERT improved:

- Accuracy by approximately **0.62 percentage points**
- Recall by approximately **1.40 percentage points**
- F1-score by approximately **0.63 percentage points**

Precision decreased slightly by approximately **0.14 percentage points**.

---

## Confusion Matrix Results

### TF-IDF + Logistic Regression

- True negatives: 1,987
- False positives: 13
- False negatives: 40
- True positives: 1,960

### DistilBERT

- True negatives: 1,984
- False positives: 16
- False negatives: 12
- True positives: 1,988

DistilBERT reduced false negatives from 40 to 12, meaning it detected 28 additional AI-generated texts. However, false positives increased from 13 to 16.

---

## Error Analysis

The project examined:

- Highly confident false positives
- Highly confident false negatives
- Errors shared by both models
- Errors made only by Logistic Regression
- Errors made only by DistilBERT
- Error rates across text-length quartiles

### Shared and Model-Specific Errors

- Errors made by both models: 12
- Baseline mistakes corrected by DistilBERT: 41
- New mistakes introduced by DistilBERT: 16

### Observed False Positives

Some human-written essays were incorrectly classified as AI-generated because they used:

- Clear argumentative structures
- Repeated transitions
- Formulaic conclusions
- Repeated phrases
- Consistent paragraph organization

One human-written essay received an AI probability of approximately **0.9997**, showing that high confidence does not guarantee correctness.

### Observed False Negatives

Some AI-generated texts were incorrectly classified as human because they included:

- Deliberate spelling mistakes
- Grammar errors
- Placeholder names
- Corrupted words
- Informal student-like phrasing
- Repeated typographical errors

These examples show that generated text can evade detection by imitating low-quality or heavily edited human writing.

---

## TF-IDF Feature Analysis

The Logistic Regression model associated some features with AI-generated text, including:

- `important`
- `additionally`
- `potential`
- `essay`
- `conclusion`
- `firstly`
- `essential`

Some human-associated features included:

- `people`
- `going`
- `car`
- `students`
- `percent`
- `paragraph`
- `school`

These are correlations in this dataset, not universal proof of authorship. Some terms may reflect dataset topics, sources, or formatting.

---

## Ethical Considerations

AI-generated text detectors should not be treated as definitive proof of academic misconduct.

### False Accusations

The baseline produced 13 false positives, while DistilBERT produced 16. Each false positive represents a human writer who could be wrongly accused.

### Non-Native English Writers

Non-native English writers may use:

- Learned academic templates
- Repeated transitions
- Translation tools
- Grammar-checking tools
- Highly structured phrasing

These patterns may resemble AI-generated text and increase the risk of unfair classification.

### Evasion

The false negatives showed that generated text can evade detection through spelling errors, grammar corruption, paraphrasing, manual editing, and informal phrasing.

### Recommended Use

The model should be used only as a screening tool.

A flagged text should trigger further human review, including:

- Draft history
- Writing samples
- Citations
- Assignment context
- Revision history
- An opportunity for the author to explain the writing process

A detector score should never be the only evidence used to penalize a student or author.

---

## Limitations

1. The dataset is balanced, while real-world class proportions may differ.
2. Results may depend on dataset-specific topics and formatting.
3. DistilBERT truncates texts longer than 256 tokens.
4. Only one Transformer architecture was evaluated.
5. Results may not generalize to newer language models.
6. Edited or paraphrased AI text may evade detection.
7. The model may rely on source-specific or topic-specific patterns.
8. High confidence does not guarantee correctness.
9. Performance may drop in another domain.
10. Hyperparameter tuning was limited by available computing resources.

---

## Requirements

Install the required libraries using:

```bash
pip install transformers datasets accelerate torch scikit-learn pandas matplotlib joblib
```

A GPU is strongly recommended for DistilBERT fine-tuning.

---

## How to Run the Project

### Google Colab

1. Open Google Colab.
2. Upload:

```text
SEA820_Final_Project_AI_Text_Detection_SUBMISSION_READY.ipynb
```

3. Upload:

```text
ai_human_sample_40000.csv
```

4. Select:

```text
Runtime → Change runtime type → T4 GPU
```

5. Run the installation cell.
6. Select:

```text
Runtime → Run all
```

### Local Jupyter Notebook

Place these files in the same folder:

```text
SEA820_Final_Project_AI_Text_Detection_SUBMISSION_READY.ipynb
ai_human_sample_40000.csv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run all cells.

---

## Expected Outputs

The notebook generates:

```text
tfidf_vectorizer.joblib
logistic_regression_baseline.joblib
best_distilbert_ai_detector/
final_model_comparison.csv
distilbert_experiment_log.csv
final_model_predictions_and_errors.csv
train_split.csv
validation_split.csv
test_split.csv
```

Large model files and datasets should not be committed to GitHub.

---

## Reproducibility

The project uses a fixed random seed:

```python
SEED = 42
```

The same train, validation, and test split is used for both models. The final test set is not used for hyperparameter selection. The best DistilBERT configuration is selected using validation F1-score.

---

## Recommended `.gitignore`

```gitignore
# Large datasets
ai_human_sample_40000.csv
AI_Human_balanced_dataset.csv
train_split.csv
validation_split.csv
test_split.csv
*.zip

# Trained models and checkpoints
*.joblib
*.bin
*.safetensors
checkpoints/
Experiment_A_checkpoints/
Experiment_B_checkpoints/
best_distilbert_ai_detector/

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc

# Logs
logs/
runs/
wandb/
```

---

## Project Deliverables

The complete project submission includes:

- Jupyter Notebook
- GitHub repository
- README
- Final report in PDF
- Presentation slides in PDF

The full dataset is not included in the repository.

---

## Dataset and License Notice

The dataset is not redistributed through this repository.

Users must download the original dataset from Kaggle and follow the dataset creator's licensing and usage conditions.

---

## Conclusion

Both models achieved strong performance on the sampled dataset.

DistilBERT was the stronger overall model, achieving **99.30% F1-score**, compared with **98.67%** for TF-IDF + Logistic Regression.

The Transformer substantially reduced false negatives but produced slightly more false positives.

The results show that AI-generated text detection can be useful as a support tool, but it should not be treated as definitive proof of authorship. Human review remains necessary because both models made confident mistakes.
