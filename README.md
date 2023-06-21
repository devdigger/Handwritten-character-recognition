# Handwritten Character Recognition

This project aims to perform handwritten character recognition using convolutional neural networks (CNNs). The model is trained on the A-Z Handwritten Data dataset to classify handwritten characters from A to Z.

## Dataset

The dataset used for training and testing is the A-Z Handwritten Data, which contains 26 classes representing the English alphabet letters. Each sample is a grayscale image of size 28x28 pixels.

## Getting Started

To run this project, follow these steps:

1. Clone the repository or download the files.

2. Set up the required dependencies:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `opencv`
   - `tensorflow`
   - `scikit-learn`

3. Open the Jupyter Notebook file `ML_Project.ipynb` in an environment with Jupyter Notebook support.

4. Execute the code cells in the notebook in sequential order to train the model, evaluate its performance, and save the trained model.

5. After training, you can use the saved model to perform predictions on new handwritten characters.

## Usage

1. Load the trained model using the `load_model` function from `tensorflow.keras.models`.

2. Preprocess the input image of a handwritten character by resizing it to 28x28 pixels and converting it to grayscale.

3. Use the preprocessed image as input to the model's `predict` function to obtain the predicted class label.

4. Display the input image along with the predicted character using OpenCV.

## Example

An example is provided in the notebook for recognizing handwritten characters. It demonstrates the process of loading the trained model, preprocessing an input image, making predictions, and displaying the results.

## License

Feel free to use the code and dataset for educational and non-commercial purposes.

## Acknowledgements

- A-Z Handwritten Data: [https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)

## Author

- GitHub: [devdigger](https://github.com/devdigger)

## Contributing

Contributions to this project are welcome. Feel free to open issues and submit pull requests to suggest improvements or fix any bugs.
