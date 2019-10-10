# digitPredicter
A program that predicts the digit that you write.

<h1>DESCRIPTION</h1>

Here is my digit classifier. I am implementing this by logistic regression.


<h6>TRAINER CLASS:</h6>
This class is to traing the model. <br>
<h6>PREDICTOR CLASS:</h6>
This class is to predict the digit by given parameters<br>
<h6>MAIN CLASS:</h6>
This class is to test Trainer and Predicter class.<br>
      
<h3>RESULTS:</h3>
Accuracy was %96 on dataset but the tones of colors in the images in dataset was a little different also there was only 5000 images which is so small compare to even 2 MB dataset. So predictions may not be well with different kind of images with digits (backaground color etc.). To overcome this, i need a bigger dataset with more realistic digits. 


<h3>How to use it: </h3>
I actually explanied what i did in the code but to summarize, you need to choose a dataset and identify the dimensions of the images. Because you need to write dimensions manually somewhere(you will see it). after you set your dataset, you just run Main.py, it will automatically save the model for you. And by using Predictor class, you can predict whenever you want. The model is saved into a csv file. 
Actually there are 10 models since you have 10 possibble output. My dataset was small and a little old (i guess). You can find a lot of datasets in internet (kaggle, mnist). 
