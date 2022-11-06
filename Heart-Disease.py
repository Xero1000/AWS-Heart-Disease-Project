#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sagemaker
import boto3
import os
import pandas as pd
import numpy as np
import io
import time
import json
import sagemaker.amazon.common as smac
from sklearn.model_selection import train_test_split
from sagemaker import image_uris


# In[2]:


# Creating the role
role = sagemaker.get_execution_role()

# Naming the bucket and getting the region
bucket_label = 'heart-disease-project-bucket'
region = boto3.session.Session().region_name

prefix = 'sagemaker/heart-disease-prediction'

# Creating the bucket
s3 = boto3.resource('s3')
try:
    if region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_label)
    else:
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
        print("Bucket creation success")
except Exception as e:
    print('S3 error:', e)


# In[3]:


# Putting the csv file from Kaggle into a dataframe
file_name = 'Heart_Disease_Prediction.csv'
hdp_dataframe = pd.read_csv('./Heart_Disease_Prediction.csv')


# In[4]:


hdp_dataframe.head()


# In[5]:


# Convert class label from string to integers. 0 = Absence, 1 = Presence
i = 0
while i < len(hdp_dataframe):
  if hdp_dataframe.at[i, 'Heart Disease'] == 'Absence':
    hdp_dataframe.at[i, 'Heart Disease'] = 0
  else:
    hdp_dataframe.at[i, 'Heart Disease'] = 1
  i = i + 1

# x = Dataframe without class label
# y = class label
x = hdp_dataframe.drop(columns=['Heart Disease'])
y = hdp_dataframe['Heart Disease']

# Splitting the data into training and testing
# x variables are the data, y variables are the class label
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.80, random_state=42)

# Splitting the training data into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=.80, random_state=42)

# Need to convert to numpy arrays otherwise we get dtype error in the next two cells
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_val = x_val.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()


# In[6]:


# Convert the training data into recordIO-wrapped protobuf format
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, x_train.astype("float32"), y_train.astype("float32"))
f.seek(0)

# Store the converted data into train folder within the bucket
boto3.Session().resource('s3').Bucket(bucket_label).Object(os.path.join(prefix, "train", "train.data")).upload_fileobj(f)


# In[7]:


# convert the validation data into recordIO-wrapped protobuf format
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, x_val.astype("float32"), y_val.astype("float32"))
f.seek(0)

# Store the converted data into validation folder within the bucket
boto3.Session().resource('s3').Bucket(bucket_label).Object(os.path.join(prefix, "validation", "validation.data")).upload_fileobj(f)


# In[8]:


# Name the training job to be created 
training_job_name = "Heart-Disease-Training-Job" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

# Identify the output location within the bucket
# Look for linear learner image uri and create container for linear learner model
output_location = 's3://{}/{}/output'.format(bucket_label, prefix)
container = image_uris.retrieve(region=boto3.Session().region_name, framework='linear-learner')

# Identify the parameters for the linear learner
linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": training_job_name,
    "AlgorithmSpecification": {"TrainingImage": container, "TrainingInputMode": "File"},
    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.c4.2xlarge", "VolumeSizeInGB": 10},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket_label, prefix),
                    "S3DataDistributionType": "ShardedByS3Key",
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None",
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket_label, prefix),
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None",
        },
    ],
    "OutputDataConfig": {"S3OutputPath": output_location},
    "HyperParameters": {
        "feature_dim": "13",
        "mini_batch_size": "100",
        "predictor_type": "binary_classifier",
        "epochs": "13",
        "num_models": "32",
        "loss": "logistic",
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 60 * 60},
}


# In[9]:


get_ipython().run_cell_magic('time', '', '\n#region = boto3.Session().region_name\nsm = boto3.client("sagemaker") # Sagemaker client\n\n# Create the the training job using the parameters from previous cell\nsm.create_training_job(**linear_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=training_job_name)["TrainingJobStatus"]\nprint(status)\nsm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=training_job_name)\nif status == "Failed":\n    message = sm.describe_training_job(TrainingJobName=training_job_name)["FailureReason"]\n    print("Training failed with the following error: {}".format(message))\n    raise Exception("Training job failed")')


# In[10]:


# Setting up the model for hosting
hosting_container = {
    "Image": container,
    "ModelDataUrl": sm.describe_training_job(TrainingJobName=training_job_name)["ModelArtifacts"][
        "S3ModelArtifacts"
    ],
}

model = sm.create_model(
    ModelName=training_job_name, ExecutionRoleArn=role, PrimaryContainer=hosting_container
)


# In[11]:


# Endpoint configurations
endpoint_config_name = "endpoint-config-" + time.strftime(
    "%Y-%m-%d-%H-%M-%S", time.gmtime()
)
print(endpoint_config_name)
endpoint_config = sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m4.xlarge",
            "InitialInstanceCount": 1,
            "ModelName": training_job_name,
            "VariantName": "AllTraffic",
        }
    ],
)


# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Creating the endpoint using the endpoint configurations from previous cell\nendpoint_name = "endpoint-" + time.strftime("%Y%m%d%H%M", time.gmtime())\nprint(endpoint_name)\nendpoint = sm.create_endpoint(\n    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name\n)\n\nresp = sm.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp["EndpointStatus"]\nprint("Status: " + status)\n\nsm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)\n\nresp = sm.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp["EndpointStatus"]\n\nprint("Status: " + status)\n\nif status != "InService":\n    raise Exception("Endpoint creation did not succeed")')


# In[13]:


# Invoking the endpoint to test it using the testing data
runtime = boto3.client("runtime.sagemaker")

# Testing data must first be converted to csv format
csv = io.BytesIO()
np.savetxt(csv, x_test, delimiter=",", fmt="%g")

payload = csv.getvalue().decode().rstrip()
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload
)
result = json.loads(response["Body"].read().decode())
predictions = np.array([r["score"] for r in result["predictions"]])


# In[14]:


# Scores in test_pred that are greater than 0.5 mean the predicted label is 1 (Presence)
# Scores less than 0.5 mean that predicted label is 0 (Absence)
prediction_labels = (predictions > 0.5)

# Multiply the mean of the results to get the accuracy percentage
accuracy = np.mean(y_test == prediction_labels) * 100

print("Accuracy:", round(accuracy, 1), "%")


# In[15]:


sm.delete_endpoint(EndpointName=endpoint_name)


# In[16]:


bucket_to_delete = boto3.resource('s3').Bucket(bucket_label)
bucket_to_delete.objects.all().delete()


# In[17]:





# In[18]:





# In[ ]:




