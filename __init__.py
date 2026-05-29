import sys
from .algo import utils
from .algo import config

sys.modules['AgentPoison.utils'] = utils
sys.modules['AgentPoison.config'] = config
