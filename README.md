# Neurabble
A neural net that will finally help you to win in the dobble boardgame!

In the ```notebooks/DobbleLabelling.ipynb``` you can see the data labelling process. To do that efficiently, I was using one of my
previous projects [ImageInspector](https://github.com/JanMalinowski/image_inspector)

Running ```sh create_data.sh``` will create two datasets. The first of them will be for multiclass classification of
objects on the dobble cards.
The other one will contain pairs of cards' images with the name of their common element.

TODO:
- Take a preatrain neural network and train it for multiclass detection of the objects on the cards.
- Having trained finetuned the network, I am going to use it to detect the common element on an image consisting of two cards.
- Deploying the app.

## License
[MIT](https://choosealicense.com/licenses/mit/)
