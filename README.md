# cs253-python-assignment

This repository contains the code and data files for the project

**Note:** The file **final.py** is the final code to be submitted and **submission.csv** is the final submitted csv on Kaggle

Below is a brief description of each file in this repository:

1. **final.py**: Contains the final code submission for the project.
2. **submission.csv**: Contains the final `.csv` file submitted on Kaggle for evaluation.
3. **validation.py**: Contains the code used for validation purposes on the train data.
4. **test.csv**: Contains the test data provided for evaluating the model.
5. **train.csv**: Contains the train data provided for model training.
6. **sample_submission.csv**: Contains the sample submission provided as a reference.
7. **code_blocks_models.py**: Contains various models and code blocks used during the development phase before arriving at the final model. The code here can be run by replacing the code in the model block of final.py or validation.py
8. **requirements.txt**: Text file containing the libraries required to run the project.

## Usage

- Use `final.py` to run the final code and generate predictions using the trained model.
- Use `validation.py` for validating the model's performance on the train data.
- The provided `.csv` files (`test.csv`, `train.csv`, `sample_submission.csv`) can be used for data preprocessing and model training.

## Requirements

Ensure you have the necessary dependencies installed to run the code successfully. You may use the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Running final.py

Follow these steps to run `final.py`:

1. Make sure you have installed the required libraries by running `pip install -r requirements.txt`.
2. Place the `train.csv` and `test.csv` files in the same directory as `final.py`.
3. Open a terminal or command prompt.
4. Navigate to the directory containing `final.py`.
5. Run the following command to execute `final.py`:
   ```bash
   python final.py
   ```
6. The script will run and generate predictions using the trained model. The output will be saved as `submission.csv`.
