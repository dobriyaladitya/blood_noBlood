# Image classification for bloody and non bloody faces
How to implement:

Install pip using,
  sudo easy_install pip
  
Install tensorflow and keras using pip
  pip install tensorflow
  pip install keras
  
Download the data given as is and make sure that you the images are seperated into two folders for training and test with about 300 images each for blood and no blood for training and 100 each for testing.
https://drive.google.com/open?id=1nw6avGM9HtMfaKocb9p-b4XMBxoVZV8Z

Download the .py file and run in python or using command from directory downloaded to
  python aditya_assignment.py

Note: The topmost comment is added in order to avoid a syntax error on newer versions of MacOS, deleting it might give an error that reads, "SyntaxError: Non-ASCII character '\xe2' in file aditya_assignment.py on line 32, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details"

Tweak the number of epochs, input_shape, filters and dropout value to play with accuracy and loss.
