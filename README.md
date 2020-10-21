# Neurabble
A neural net that will finally help you to win in the dobble boardgame!

In the ```notebooks/DobbleLabelling``` you can see the data labelling process.

Running the script ```create_data.sh``` will create two datasets. The first of them will be for multiclass classification of
objects on the dobble cards.
The other one will contain pairs of cards' images with the name of their common element.

TODO:
- Take a preatrain neural network and train it for multiclass detection of the objects on the cards.
- Having trained finetuned the network, I am going to use it to detect the common element on an image consisting of two cards.
- Deploying the app.
