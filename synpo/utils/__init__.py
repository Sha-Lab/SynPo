from .config import *
from .trainer import *
from .tf_logger import Logger
from .utils import *
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('MAIN')
logger.setLevel(logging.INFO)
