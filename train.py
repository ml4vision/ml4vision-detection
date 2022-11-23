from engine import Engine
from configs.config import get_config
from ml4vision.client import Client, Project

API_KEY = 'API_KEY'
PROJECT_NAME = 'PROJECT_NAME'
PROJECT_OWNER = None

# init client
client = Client(API_KEY)

# load project
project = Project.get_by_name(client, PROJECT_NAME, owner=PROJECT_OWNER)
project_location = project.pull(location='./data')

# get config
config = get_config(project_location=project_location, categories=project.categories)

# train model
engine = Engine(config)
engine.train()

# upload model
engine.upload(project)