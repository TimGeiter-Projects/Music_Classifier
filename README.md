# Music Genre Classifier
Es wurde ein Modell entwickelt und trainiert, das Musikstücke automatisiert den Genres Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop und Rock zuordnet. 

Im Rahmen der Arbeit wurden zwei unterschiedliche Ansätze umgesetzt:
1. Eine Klassifikation auf Basis explizit definierter Merkmale, bei der vorab ausgewählte Eigenschaften der Musikstücke zur automatischen Einordnung herangezogen werden, sowie
2. eine Klassifikation mittels eines Convolutional Neural Networks (CNN), bei der das Modell relevante Merkmale eigenständig während des Trainingsprozesses identifiziert und nutzt.
## Dataset
Es wurde der Datensatz Free Music Archive (FMA) (https://github.com/mdeff/fma?tab=readme-ov-file)  verwendet. Konkret wurde der „fma_small“ Datensatz mit 8 balancierten
Musikklassen (1000 Tracks pro Klasse) betrachtet. 
## Ergebnisse Ansatz 1.
![Training_History](confusion_matrix_segments.png)
## Test Accuracy: 73.42%

## Ergebnisse Ansatz 2.
### Training Curves
![Training History](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix_cnn.png)

## Test Accuracy: 62.39%
```

