# Spam-Classifier
The spam classifier is a machine learning model designed to distinguish between spam and non-spam (ham) messages.

```markdown
# Text Classification with Word Vectors

This project focuses on text classification using word vectors. The provided code implements a TextClassifier class that preprocesses data, generates word vectors, and trains various machine learning models for text classification.

## Dataset

The dataset used for this project is located in 'C:\\Users\\yjun0\\OneDrive - Asia Pacific University\\A.P.U\\Projects\\spam_ham_dataset.csv'. It consists of text messages labeled as 'ham' or 'spam'.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install matplotlib pandas numpy nltk wordcloud scikit-learn
   ```

3. Run the provided example usage code in a Python environment:
   ```python
   python example_usage.py
   ```

## Class Structure

### TextClassifier

#### Methods

- `__init__(self, data_path)`: Initializes the TextClassifier with the provided dataset path.
- `preprocess_data(self)`: Preprocesses the dataset by replacing labels and removing stopwords and punctuation.
- `generate_wordcloud(self, text, color)`: Generates and displays a word cloud for the given text and color.
- `build_word_vectors(self)`: Builds word vectors based on the dataset.
- `train_predict_models(self, features, targets)`: Trains and evaluates multiple machine learning models using the provided features and targets.

## Evaluation Results

The evaluation results on the validation and test sets are as follows:

```python
[('SVC', 0.7306701030927835, 0.7255154639175257),
 ('KN', 0.7306701030927835, 0.7255154639175257),
 ('NB', 0.7306701030927835, 0.7255154639175257),
 ('DT', 0.7306701030927835, 0.7255154639175257),
 ('LR', 0.7306701030927835, 0.7255154639175257),
 ('RF', 0.7306701030927835, 0.7255154639175257)]
```

These results indicate the validation and test accuracies for different classifiers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Replace `<repository-url>` and `<repository-folder>` with the appropriate values for your project. You may also want to include information about how to reproduce the results and any additional details about the project or its dependencies.
