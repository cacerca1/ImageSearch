This repository contains the code necessary to perform content based image retrieval. 
The process is first tested locally in Local_testing.ipynb, then scaled up by using AWS services. 

The main procedure is as follows: 
1. Build a feature extraction model
2. Infer on all the images in the set and store the features
3. Load the features into OpenSearch (ElasticSearch) 
4. Build an inference endpoint using Amazon Sagemaker
5. Build a Lambda function to perform inference and KNN search given an input image
6. Setup a REST API using API Gateway to manage accesssing the lambda function

From here, the user can make a request to the api with a base64 encoded image and a value for K, the number of requested similar images. 

In this repo, a simple frontend is used to manage the query image selection, rest api request, and result visualizaiton. 

![Sample result](examples/frontend.png?raw=true "Title")
