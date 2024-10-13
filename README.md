# Usizo



Usizo is a project to help improve access to cheap credit to Africans in Sub-Saharan Africa. To do this, I aim to build an ML-based decision engine for extending credit. The first phase of this process, which I am doing in this notebook, is training a model to serve as a decision engine. This model will use publicly-available data to train a proof-of-concept model. If successful, then this project will be replicated using African data.



## Data

The data used in this task was obtained uploaded to OpenML by Dustin Carrion. The link to the dataset is here: https://www.openml.org/search?type=data&sort=runs&id=43454&status=active.



### Structure

The data contains the following columns:



- person_age (int): The age of the borrower

- person_income (int): How much the person makes annually in euros.

- person_home_ownership (enum): Either rent, own, mortgage, or

- person_emp_length (float): How long the persn has been employed (in years)

- loan_intent (enum): The intended purpose of the borrrowed funds.

- loan_grade (enum): The grade of the loan. This is better explained [here](https://www.thebalancemoney.com/what-is-loan-grading-5211003).

- loan_amnt (float): The amount borrowed.

- loan_int_rate (float): The loan interest rate.

- loan_status (enum): The present status of the loan where 0 means they have not defaulted and 1 means they have defaulted.

- loanpercentincome Percent income

- cb_person_default_on_file (enum): Either 0 or 1 where 1 means the borrower has defaulted on loans in the past and 0 means they have not.

- cb_preson_cred_hist_length (float): How long the person has been using credit in years.



### Quantity

There are 32,581 rows in the dataset which should be plenty enough for training.


# Step 0: Set Up Environment




## Step 0.1: Installing additional libraries

The data is stored using arff file format. `liac-arff` is a good library for reading .arff files.

## Step 0.2: Importing required libraries



- NumPy for numerical processing

- Arff for reading the data

- Pandas for managing the data as a dataframe

- Tensorflow for building and training the model

- Sklearn for preprocessing and splitting the data

- Matplotlib and Seaborn for plotting diagrams

# Step 1: Reading and Preprocessing the Data



During this step, I will read the data and prepare it for training. Preparations will include cleaning up the data, selecting the features and splitting the data into training and test data.

### Step 1.1: Reading the data

### Step 1.2: Inspecting the data



During this stage, I identified several issues with the data.



1. Two columns(`loan_int_rate` and `person_emp_length`) are missing values. I will fill these columns with mean values.



2. Certain age values are above 140. This is likely to be an error within the dataset. I'm going to drop all values above 90.





3. The target classes(`loan_status`) are imbalanced with over 80% of the values belonging to class 0(not default). To correct this, I will use oversampling.

## Step 1.3: Preprocessing the data



During this process, I did the following to prepare the data for training:

1. I cleaned the data by filtering rows that had ages above 90.

2. I encoded the data so that it is entirely numeric.

3. I filled missing values by the mean of that feature and class.

4. I oversampled the class 1 to balance the classes.

5. Explored features to identify the most relevant and selected them for training.

### Step 1.3.1: Filtering rows with age above 90

### Step 1.3.2: Encoding

### Step 1.3.3: Filling Missing values

### Step 1.3.4: Balancing the classes

### Step 1.3.5: Selecting features



Based on the heatmap shown below, the most features that indicate a strong correlation with the target(`loan_status`) are:

1. person_income

2. person_home_ownership

3. loan_grade

4. loan_amt

5. loan_int_rate

6. loan_percent_income

7. cb_person_default_on_file



Now consider the following,

1. loan_percent_income = loan_amt / person_income. Therefore if we consider only loan_percent_income, we would have considered both loan_amt and person_income.

2. loan_grade is strongly correlated with loan_int_rate. Which makes sense as loan_grade is based on perceived risk and the loan_int_rate is also based on the same perceived risk. Therefore it should suffice to use only one of those features. I will use loan_int_rate as it is more granular.



This leaves us with the following four features:

1. person_home_ownership

2. loan_int_rate

3. loan_percent_income

4. cb_person_default_on_file

# Step 2: Building vanilla model

## Step 2.1: Defining the model architecture



I settled for the 16, 8, 1 model architecture. I started with the (8, 4, 1) and it produced an accuracy of around 74%. I experimented with a deeper network with more layers(32, 16, 8, 4, 1)) and accuracy dropped dramatically to 50%. I then experimented with a wider network (64, 16, 4, 1) but it did not perform better than the initial (8, 4, 1) architecture. Eventually, I decided on the simpler (8, 4, 1) as the most optimal architecture that was smaller and produced reasonable accuracy.

## Step 2.2: Compiling the model.



For optimizers, I compared rmsprop, adam and sgd. Of all three, adam gave the best accuracy.

Step 2.3: Training the model

For training I tried different batch sizes, 32, 4096 and 2048. There were no material differences in the accuracies but 32 gave marginally better binary_accuracy during training. However, it trained slower than the other batch sizes. I decided to increase the batch size to 64.



The model seemed to converge quickly to about 75% accuracy. And there were no significant improvements from that point onwards. Therefore I reduced the number of epochs to 50 from the initial 200.

# Step 3: Building the Optimised model



With these ideas, I will be implementing the network with several optimisation techniques that I will tune to improve model performance. The technqiues will include:

1. Regularisation

2. Early Stopping

3. Learning Rate Adjustment

4. And Dropout layers

## Step 3.1: Defining the model



From the previous model, I added dropout and regularisers.



I started with a dropout rate of 0.5 and that produced an accuracy of about 70% during training. It seemed the model was not picking on enough trends about the data. So I decided to reduce it to 0.2. This improved the training accuracy to about 78%. Reducing it to 0.1 did not change the performance much.



This also experimented with l2 regularizer by reducing it a bit and the performance improved marginally.



I also experimented with the model architecture, eventually settling for (32, 16, 1) as the most optimal. Wider or deeper networks led to poor performance.

## Compiling the model



I also tried different optimisers, none of them worked as well as adam. sgd dropped the accuracy to 70%.

## Training

Early stopping helped the model stop once the most optimal weights had been found.

# Error Analysis

## Vanilla Model



The Vanilla Model registers and accuracy of 0.77.

## Optimised Model



The Optimised Model performs better with an accuracy of 0.79.

# Saving the models
