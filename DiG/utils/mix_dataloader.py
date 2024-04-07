import utils
from dataset.dataset_lmdb import ImageLmdb
from dataset.dataset_word import WordPkl

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()

image_dataset = ImageLmdb()
word_dataset  = WordPkl()