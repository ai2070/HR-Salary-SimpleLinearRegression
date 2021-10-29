#!/usr/bin/env python
# coding: utf-8

# # 1.UNDERSTAND THE PROBLEM STATEMENT 

# - The objective of this case study is to predict the employee salary based on the number of years of experience. 
# - In simple linear regression, we predict the value of one variable Y based on another variable X.
# - X is called the independent variable and Y is called the dependant variable.
# - Why simple? Because it examines relationship between two variables only.
# - Why linear? when the independent variable increases (or decreases), the dependent variable increases (or decreases) in a linear fashion.
# 

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[118]:


# install seaborn library
get_ipython().system('pip install seaborn')

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[119]:


# read the csv file 
salary_df = pd.read_csv('salary.csv')
salary_df


# In[120]:


salary_df.head(7)


# In[121]:


salary_df.tail(7)


# In[122]:


salary_df['Salary'].max()


# # 3. PERFORM EXPLORATORY DATA ANALYSIS AND VISUALIZATION

# In[123]:


# check if there are any Null values
sns.heatmap(salary_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[124]:


# Check the dataframe info
salary_df.info()


# In[125]:


# Statistical summary of the dataframe
salary_df.describe()


# In[126]:


# number of years of experience corresponding to employees with minimum and maximim salaries
max = salary_df[salary_df['Salary'] == salary_df['Salary'].max()]
max


# In[127]:


min = salary_df[salary_df['Salary'] == salary_df['Salary'].min()]
min


# In[128]:


salary_df.hist(bins = 30, figsize = (20,10), color = 'r')


# In[129]:


# plot pairplot
sns.pairplot(salary_df)


# In[130]:


corr_matrix = salary_df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()


# In[131]:


sns.regplot(x="YearsExperience", y="Salary", data=salary_df);


# # 4. CREATE TRAINING AND TESTING DATASET

# In[132]:


#Set X and Y (target and depended variable)
X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]


# In[133]:


X


# In[134]:


y


# In[135]:


X.shape


# In[136]:


y.shape


# In[137]:


# Let's create an array
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')


# In[138]:


# Only take the numerical variables and scale them
X


# In[139]:


# split the data into test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[140]:


#data are now shuffled


# In[141]:


X.shape, y.shape


# In[142]:


X_test.shape , y_test.shape


# # 5. TRAIN A LINEAR REGRESSION MODEL IN SK-LEARN

# In[143]:


# using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regresssion_model_sklearn = LinearRegression(fit_intercept = True)
regresssion_model_sklearn.fit(X_train, y_train)


# In[144]:


regresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test, y_test)
regresssion_model_sklearn_accuracy


# In[145]:


print('Linear Model Coefficient (m): ', regresssion_model_sklearn.coef_)
print('Linear Model Coefficient (b): ', regresssion_model_sklearn.intercept_)


# # 6. EVALUATE TRAINED MODEL PERFORMANCE

# In[146]:


y_predict = regresssion_model_sklearn.predict(X_test)


# In[147]:


y_predict


# In[148]:


plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regresssion_model_sklearn.predict(X_train), color = 'red')
plt.ylabel('Salary')
plt.xlabel('Number of Years of Experience')
plt.title('Salary vs. Years of Experience')


# In[161]:


# Let's preditct salary of an employee with 5 years of experience
num_years_exp = [[5]]
salary = regresssion_model_sklearn.predict(num_years_exp)
salary


# An employee with 5 years' experience will get 7,000.00 dollars salary

# # 7. TRAIN A LINEAR LEARNER MODEL USING SAGEMAKER BUILT-IN FEATURES

# In[165]:


# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2

import sagemaker
import boto3

# Let's create a Sagemaker session
sagemaker_session = sagemaker.Session()

# Let's define the S3 bucket and prefix that we want to use in this session
bucket = 'sagemaker-practical' # bucket named 'sagemaker-practical' was created beforehand
prefix = 'linear_learner' # prefix is the subfolder within the bucket.

# Let's get the execution role for the notebook instance. 
# This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.
# Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, from the S3 bucket and writing training results to Amazon S3). 
role = sagemaker.get_execution_role()
print(role)


# In[166]:


X_train.shape


# In[167]:


y_train = y_train[:,0]


# In[168]:


y_train.shape


# In[85]:


import io # The io module allows for dealing with various types of I/O (text I/O, binary I/O and raw I/O). 
import numpy as np
import sagemaker.amazon.common as smac # sagemaker common libary

# Code below converts the data in numpy array format to RecordIO format
# This is the format required by Sagemaker Linear Learner 

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0) 
# When you write to in-memory byte arrays, it increments 1 every time you write to it
# Let's reset that back to zero 


# In[86]:


import os

# Code to upload RecordIO data to S3
 
# Key refers to the name of the file    
key = 'linear-train-data'

# The following code uploads the data in record-io format to S3 bucket to be accessed later for training
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

# Let's print out the training data location in s3
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[87]:


X_test.shape


# In[88]:


y_test.shape


# In[ ]:


# Make sure that the target label is a vector
y_test = y_test[:,0]


# In[90]:


# Code to upload RecordIO data to S3

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_test, y_test)
buf.seek(0) 
# When you write to in-memory byte arrays, it increments 1 every time you write to it
# Let's reset that back to zero 


# In[91]:


# Key refers to the name of the file    
key = 'linear-test-data'

# The following code uploads the data in record-io format to S3 bucket to be accessed later for training
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)

# Let's print out the testing data location in s3
s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_test_data))


# In[92]:


# create an output placeholder in S3 bucket to store the linear learner output

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))


# In[93]:


# This code is used to get the training container of sagemaker built-in algorithms
# all we have to do is to specify the name of the algorithm, that we want to use

# Let's obtain a reference to the linearLearner container image
# Note that all regression models are named estimators
# You don't have to specify (hardcode) the region, get_image_uri will get the current region name using boto3.Session


from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'linear-learner')


# In[95]:


# We have pass in the container, the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)


# We can tune parameters like the number of features that we are passing in, type of predictor like 'regressor' or 'classifier', mini batch size, epochs
# Train 32 different versions of the model and will get the best out of them (built-in parameters optimization!)

linear.set_hyperparameters(feature_dim = 1,
                           predictor_type = 'regressor',
                           mini_batch_size = 5,
                           epochs = 5,
                           num_models = 32,
                           loss = 'absolute_loss')

# Now we are ready to pass in the training data from S3 to train the linear learner model

linear.fit({'train': s3_train_data})

# Let's see the progress using cloudwatch logs


# MINI CHALLENGE
# - Try to train the model with more epochs and additional number of models
# - Can you try to reduce the cost of billable seconds?

# In[ ]:





# # TASK #8: DEPLOY AND TEST THE TRAINED LINEAR LEARNER MODEL 

# In[39]:


# Deploying the model to perform inference 

linear_regressor = linear.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')


# In[40]:


from sagemaker.predictor import csv_serializer, json_deserializer

# Content type overrides the data that will be passed to the deployed model, since the deployed model expects data in text/csv format.

# Serializer accepts a single argument, the input data, and returns a sequence of bytes in the specified content type

# Deserializer accepts two arguments, the result data and the response content type, and return a sequence of bytes in the specified content type.

# Reference: https://sagemaker.readthedocs.io/en/stable/predictors.html

linear_regressor.content_type = 'text/csv'
linear_regressor.serializer = csv_serializer
linear_regressor.deserializer = json_deserializer


# In[41]:


# making prediction on the test data

result = linear_regressor.predict(X_test)


# In[42]:


result # results are in Json format


# In[43]:


# Since the result is in json format, we access the scores by iterating through the scores in the predictions

predictions = np.array([r['score'] for r in result['predictions']])


# In[44]:


predictions


# In[45]:


predictions.shape


# In[46]:


# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, predictions, color = 'red')
plt.xlabel('Years of Experience (Testing Dataset)')
plt.ylabel('salary')
plt.title('Salary vs. Years of Experience')


# In[47]:


# Delete the end-point

linear_regressor.delete_endpoint()


# # EXCELLENT JOB! NOW YOU'RE FAMILIAR WITH SAGEMAKER LINEAR LEARNER, YOU SHOULD BE PROUD OF YOUR NEWLY ACQUIRED SKILLS
