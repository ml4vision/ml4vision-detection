from engine import Engine
from configs.config import get_config
from ml4vision.client import Client

API_KEY = '8bf9cd1d7a1c130e206a8434a5bc89d080e61ea7'
PROJECT_NAME = 'Pollen counting (CPD-1)'
PROJECT_OWNER = None

# init client
client = Client(API_KEY)

# load project
project = client.get_project_by_name(PROJECT_NAME, owner=PROJECT_OWNER)
project_location = project.pull(location='./data')

# get config
config = get_config(project_location=project_location, categories=project.categories)
config.solver.max_epochs = 2

# train model
engine = Engine(config)
engine.train()

# upload model
# engine.upload(project)