import json
from os import environ
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from io import BytesIO
import logging
from urllib.parse import urlparse
import numpy as np
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Global variables that are reused
sm_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

EMBEDDING_MODEL_ENDPOINT_NAME = environ['sagemaker_model']
ES_ENDPOINT = environ['es_endpoint']
ES_USER = environ['es_user']
ES_SECRET = environ['es_secret']
ES_INDEX = environ['es_index']

def connect_to_ES(esEndPoint):
    """Connet to ElasticSearch using prescribed endpoint

    Args:
        esEndPoint (str): endpoint for ElasticSearch service.

    Returns:
        ElasticSearch: client to ES
    """
    print ('Connecting to the ES Endpoint {0}'.format(esEndPoint))
    try:
        esClient = Elasticsearch(
            hosts=[{'host': ES_ENDPOINT, 'port': 443}],
            http_auth=(ES_USER, ES_SECRET),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection)
        return esClient
    except Exception as E:
        print("Unable to connect to {0}".format(esEndPoint))
        print(E)
        
def get_predictions(payload): 
    """Get predictions from sagemaker endpoint.

    Args:
        payload (dict): Dictionary with inputs to sagemaker endpoint. Dict has to have 'inputs' key.

    Returns:
        dict: dictionary with model embeddings of payload.
    """
    return sm_client.invoke_endpoint(EndpointName=EMBEDDING_MODEL_ENDPOINT_NAME, 
                                     Body=payload,
                                     ContentType='application/json') 

def extract_features(encoded_image): 
    """[summary]

    Args:
        encoded_image (str): string representation of image.

    Returns:
        list: list of embeddings for the image.
    """

    # get predictions
    response = get_predictions(json.dumps({'inputs':encoded_image})) 
    # get response
    response_body = json.loads((response['Body'].read())) 
    # reshape to list of len=2048
    feature_lst = list(np.array(response_body).reshape(-1))
    
    return feature_lst
    
def search_es(es, query_embeddings, k):
    """Search ES index for k closest elements to the query embeddings passed in.

    Args:
        es (ElasticSearch): Instance of elastic search.
        query_embeddings (list): embeddings of image to search against.
        k (int): Number of resutls requested

    Returns:
        [list]: s3 objects returned by the ES query.
    """
    es_query ={
                "query": {
                    "knn": {
                        "embeddings": {
                            "vector": query_embeddings,
                            "k": k
                        }
                    }
                }
        }
    
    res = es.search(index=ES_INDEX, body=es_query, size=k)
    uris = [hit['_source']['uri'] for hit in res['hits']['hits']]
    return uris
    
def create_presigned_url(uris, expiration=3600):
    """Sample function from boto3.amazonaws.com
    
    Generate a presigned URL to share an S3 object

    :param uri: string 
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    presigned_uris = []
    try:
        for uri in uris:
    
            o = urlparse(uri, allow_fragments=False)
    
            response = s3_client.generate_presigned_url('get_object',
                                                        Params={'Bucket': o.netloc,
                                                                'Key': o.path.lstrip('/')},
                                                        ExpiresIn=expiration)
            presigned_uris.append(response)
            
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return presigned_uris
    
def lambda_handler(event, context):
    """Main function for lambda based Image Query

    Returns:
        [list]: list of presigned uris to closest images found in ES.
    """
    # get the image data
    if 'queryStringParameters' in event:
        image = event['queryStringParameters']['base64image']
        k = event['queryStringParameters']['k'] # number of similar images

    else:
        image = event['base64image']
        k = event['k'] # number of similar images
        
    # connect to elastic search
    es = connect_to_ES(ES_ENDPOINT)

    # get the image features from the sagemaker endpoint
    features = extract_features(image)
    uris = search_es(es, features, k)
    presigned_uris = create_presigned_url(uris, expiration=3600*24*2)

    return {
        'statusCode': 200,
        'headers':{'Content-Type':'application/json'},
        'body': json.dumps({'images': presigned_uris})
    }
