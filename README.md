# college-inquiry-chatbot
before running the project first you have to download 
in intent1.json we are creating our dataset for chatbot
you can add further data in intent1

you can use any terminal to download this files which is required to run chatbot
python -m pip install –upgrade pip
pip install nltk
pip install pandas
pip install tflearn 
pip install tensorflow
pip install pickle-mixin
pip install PyAudio
pip install SpeachRecognition

after install all this you can run 
python app.py 

this file will load the data and train our data and create data.pickle file which work as our train model

after this have to run app_test.py
python app_test.py
this file will use data.pickle file as train model and generate the output

by using the speach recognition library it will convert your word into text it produced output. 
