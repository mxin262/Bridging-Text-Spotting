import lmdb
import pickle
from tqdm import tqdm

# args to load lmdb
data_paths = ['/home/ymk-wh/workspace/datasets/text_recognition/CVPR2016',
              '/home/ymk-wh/workspace/datasets/text_recognition/NIPS2014']
voc_set = set()

# start data loadding
for data_path in data_paths:
  env = lmdb.open(data_path, max_readers=32, readonly=True)
  txn = env.begin()

  num_samples = int(txn.get(b"num-samples"))
  for i in tqdm(range(num_samples)):
    index = i + 1
    word_key = b'label-%09d' % index
    word = txn.get(word_key).decode()
    voc_set.add(word)

# save vocabulary word to disk
voc_txt_path = '/home/ymk-wh/workspace/datasets/text_recognition/synth_voc.pkl'
with open(voc_txt_path, 'wb') as f:
  pickle.dump(voc_set, f)