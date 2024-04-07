from __future__ import absolute_import

from .metrics import Accuracy, CTCAccuracy, EditDistance, RecPostProcess, Accuracy_with_lexicon, EditDistance_with_lexicon, recognition_f_measure
from .multi_label_metrics import multi_label_f_measure


__factory = {
  'accuracy': Accuracy,
  'ctc_accuracy': CTCAccuracy,
  'editdistance': EditDistance,
  'accuracy_with_lexicon': Accuracy_with_lexicon,
  'editdistance_with_lexicon': EditDistance_with_lexicon,
  'multi_label_fmeasure': multi_label_f_measure,
  'recognition_fmeasure': recognition_f_measure,
}

def names():
  return sorted(__factory.keys())

def factory():
  return __factory