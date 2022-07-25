import os
import json
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.cloud import artifactregistry_v1

#variables
PATH_TO_KEY=os.environ.get('GOOGLE_SERVICE_ACCOUNT')          #path to json key file
PROJECT='codeday-355723'                                  #project name
LOCATION='us-west2'                                             #region of your gcp project
GAR_REPO='nano'                                               #name of google artifact registry repository
DOCKER_IMAGE='dalle-nano:latest'                                #name of image in gar repository
MODEL_DISPLAY_NAME='beta'                                       #name of model once imported
PREDICT_ROUTE='/prediction'                                     #predict route for docker image
HEALTH_ROUTE='/health'                                          #health route for docker image
PORTS=[80]                                                      #exposed docker image ports
ENDPOINT_MACHINE_TYPE='n1-standard-2'                           #machine type for endpoint

#all other parameters can be edited manually in the code body

json_acct_info = json.loads(os.environ.get('GOOGLE_SERVICE_ACCOUNT'))

my_credentials = service_account.Credentials.from_service_account_info(
    json_acct_info)

scoped_credentials = my_credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])

aiplatform.init(
    project=PROJECT,
    location=LOCATION,
    credentials=my_credentials
)

model=aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    serving_container_image_uri= f'{LOCATION}-docker.pkg.dev/{PROJECT}/{GAR_REPO}/{DOCKER_IMAGE}',
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=PORTS
)

endpoint=model.deploy(machine_type=ENDPOINT_MACHINE_TYPE,
    min_replica_count=1
)
