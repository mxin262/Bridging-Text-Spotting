from __future__ import absolute_import
import os
import sys
import numpy as np
import scipy.misc
import errno

try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x

try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
except:
  run_on_remote = False


def mkdir_if_missing(dir_path):
  try:
    os.makedirs(dir_path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      if run_on_remote:
        dir_name = os.path.dirname(fpath)
        if not mox.file.exists(dir_name):
          mox.file.make_dirs(dir_name)
          print('=> making dir ', dir_name)
        self.file = mox.file.File(fpath, 'w')
      else:
        mkdir_if_missing(os.path.dirname(fpath))
        self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
      self.file.flush()
      if not run_on_remote:
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
      self.file.close()
