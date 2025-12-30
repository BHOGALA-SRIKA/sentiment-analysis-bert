# sentiment-analysis-bert
Project Overview
This project demonstrates sentiment analysis on the IMDB movie reviews dataset using:
- A Logistic Regression baseline for comparison.
- A fine‑tuned BERT model from Hugging Face Transformers.

The goal is to classify reviews as positive or negative, and to understand how modern NLP models improve over traditional baselines.


Tech Stack
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit‑learn



Dataset
- IMDB Movie Reviews Dataset  
  [Kaggle Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
  50,000 reviews labeled as positive or negative.



Steps
1. Preprocessing
   - Clean text (remove punctuation, lowercase).
   - Tokenize using BERT tokenizer.

2. Baseline Model
   - Logistic Regression with TF‑IDF features.
   - Evaluate accuracy and F1 score.

3. Fine‑tuned BERT
   - Load bert-base-uncased from Hugging Face.
   - Fine‑tune on IMDB dataset.
   - Evaluate with accuracy and F1 score.



Results
| Model                   | Accuracy | F1 Score |
|-------------------------|----------|----------|
| Logistic Regression     | 82.0%    | 81.5%    |
| Fine‑tuned BERT         | 90.2%    | 89.7%    |



 Evaluation
- Metrics: Accuracy, F1 Score.  
- Observed that BERT significantly outperforms the baseline by capturing contextual meaning.



Challenges
- Handling large dataset size (training time).  
- Managing GPU/Colab resources.  
- Hyperparameter tuning for better performance.



How to Run
bash
pip install transformers datasets torch scikit-learn
python sentiment_analysis.py


Or open the notebook in Google Colab and run step by step.



Proof of Work
- Notebook: [Click here to view the Project Notebook]
[To Open in Colab!] https://colab.research.google.com/drive/1Uf6Qlqzs5ucsrPcbC7xldyvCpJRw--Eo?usp=sharing

- Blog: https://medium.com/@srikab2007/fine-tuning-bert-for-sentiment-analysis-8e8863837aa8
- Demo: Example prediction:  
  - Input: “The movie was absolutely fantastic!”  
  - Output: Positive
