# Text Sentience Classification

This project focuses on building an AI model capable of classifying movie reviews or comments into positive or negative sentiment categories. The model leverages both traditional machine learning and deep learning techniques, specifically Logistic Regression and LSTM (Long Short-Term Memory) networks, to perform sentiment analysis on text data.



- The dataset consists of movie reviews gathered from the IMDb dataset. Reviews are preprocessed to remove noise and tokenize the text for further analysis.
- A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used to convert raw text data into numerical form, enabling the model to interpret and classify text.

- A Logistic Regression model is trained on the processed text data to classify reviews into two categories: positive and negative. The model uses coefficients learned during training to predict the sentiment of unseen data.
- The model's performance is evaluated based on accuracy and classification reports, with additional hyperparameter tuning to improve its predictive capabilities.

-  A LSTM model, a type of Recurrent Neural Network (RNN), is also trained to handle sequences of text for better capturing of temporal dependencies and context. This helps in understanding the relationship between words and phrases that traditional models might miss.
- The LSTM model is trained using Keras, and predictions are made on the text after preprocessing with the tokenizer.

- Both models are evaluated using accuracy scores and classification reports to assess their performance. Precision, recall, and F1 scores provide insights into the models' ability to differentiate between positive and negative sentiments.

- The project includes a simple text input interface for users to test the models. The user can input text, and the system will predict whether the sentiment is positive or negative.




## Authors

- [@SabatoLaManna](https://github.com/SabatoLaManna)


## License
[![Mozilla License](https://img.shields.io/badge/License-CC_BY_NC_4.0-green)](https://www.creativecommons.org/licenses/by-nc/4.0/legalcode.en)

This project was created by Sabato La Manna, licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.


## Installation

Install Visual Studio Code
Ensure you have VS Code installed. If not, [download it here.](https://code.visualstudio.com/)
Install Python
Download and install Python (if you haven't already) from [python.org.](https://www.python.org/) I used python 3.11

Open your terminal and run the following commands to install the libraries you'll use:
```bash
pip install numpy
pip install pandas
pip install joblib
pip install scikit-learn
pip install tensorflow
pip install nltk
pip install matplotlib  # Optional, for data visualization
pip install seaborn     # Optional, for enhanced visualization
```

After installing your libraries, unzip the .tar.gz file as this contains your dataset
```bash 
tar -xvzf Dataset.tar.gz
```

Then just decide which type of model to create, for a Deep Learning model run the ```__Model_Trainer_DeepLearning.py``` file, for traditional machine learning run the ```__Model_Trainer_Traditional.py``` file. 

For usage open the ```__Test.py``` file and run it, then just use the console to type in your text. 
## Acknowledgements

 - [Large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)(Was used to train the model, also included in the files)

