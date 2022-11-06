import json

import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    data = json.loads(json.dumps(event))
    payload = data['data']
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)

    result = json.loads(response['Body'].read().decode())

    pred = int(result['predictions'][0]['predicted_label'])
    predicted_label = ''
    if pred == 1:
        predicted_label = 'Presence'
    else:
        predicted_label = 'Absence'
    
    return predicted_label
