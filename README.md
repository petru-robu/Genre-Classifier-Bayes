# Naive Bayes Genre Classifier
This project attempts to solve the genre classification problem by using a Multinomial Naive Bayes Classifier.

## Docs: [Genre Classifier Bayes](./Genre_Classifer_Bayes.pdf)

# Running instructions
## Data
The classifier uses the GTZAN Music Genre Dataset. The data is split in two halves: `half1.csv` and `half2.csv`.
The model can be trained on a half and tested on the other.

GTZAN Music Genre Dataset: [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Train
```bash
# First, you should train the model on the data set:
python3 trainer.py
```
## Predict

```bash
# You can use the trained model to predict data stored in csv files available in the data folder with:
python3 predictcsv.py

# You can use the trained model to predict data stored in wav / mp3 files available in the audio folder with:
python3 predictaudio.py
```