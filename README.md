# Content Based Image Retrieval
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

#### Deployment Architecture
![Architecture](examples/architecture.png?raw=true "Title")

## UI Example
As a minimal viable product, a simple frontend is used here to manage the query image selection, rest api request, and result visualizaiton. Users can also define their own frontends and link to the AWS rest endpoint. 

![Sample result](examples/frontend.png?raw=true "Title")

## Approaches

Three similar approaches were used for feature extraction. The first and simplest approach uses a imagenet pretrained ResNet50 as a feature extractor. This method alone works very well with minimal setup, returning embedding vectors of size 2048. 
The second approach again uses an imagenet pretrained ResNet50, but it extends this base model by adding three extra dense layers which reduce the resulting vector down to length 256. The extended network is then trained with image triplets consisting of an anchor sample, a positive sample, and a negative sample. The training procedure tries to maxize the separation between the anchor-postive and anchor-negative distance. 
The final approach is a two step process. First, a classifier network is trained to classify incoming images into one of the possible classes. Then, search is conducted only on the ResNet50 features within that class. For this approach, a second KNN classifier was trained. See the figure below for its per class performance.

![Confusion Matrix](examples/classifierConfusionMatrix.png?raw=true "Title")

## Model Comparison
There are two main factors to take into account when evaluating model perfomrnace:

1. Model accuracy
2. Speed

High accuracy will allow us to choose the best image for a given query, while low runtime will lead to better user experience as it will minimize the time required for users to get their results back. Hence, together these two metrics will dictate customer satisfaction. 
#### Model Performance
For the models described above, the accuracy for values of K ranging from 1-6 is as shown in the following figure. The result show that for the single step process, accuracy decreases as K is increased. While for the two step model, accuracy is pinned at the classifier accuracy. This is as expected since our metric of accuracy in this problem is whether the class of the predicted image matches the class of our query image.

&nbsp; | ResNet50 <br /> 2048 | ResNet50 Triplet <br /> 256 | KNN + ResNet50 <br /> 2048 | KNN + ResNet50 + PCA <br /> 2048 |
--- | --- | --- | --- |--- 
1 | 1.0   | 1.0   | 0.919 | None |
2 | 0.955 | 0.953 | 0.919 | None |
3 | 0.933 | 0.936 | 0.919 | None |
4 | 0.919 | 0.927 | 0.919 | None |
5 | 0.909 | 0.921 | 0.919 | None |
6 | 0.901 | 0.918 | 0.919 | None |

#### Time Complexity
Looking at runtime for the different models, we can that minimizing the search space reduces the runtime of the process. This can also be seen in the PCA number of components anlysis in the next table. 

| ResNet50 <br /> 2048 | ResNet50 Triplet <br /> 256 | KNN + ResNet50 <br /> 2048 | KNN + ResNet50 + PCA <br /> 2048 |
| --- | --- | --- |--- 
0.175 | 0.110 | 0.0804 | None