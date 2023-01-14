from collections import Counter, defaultdict
import itertools
from operator import itemgetter
from pathlib import Path
import pickle
from contextlib import closing

BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        # todo: client GCP
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
    
    def write(self, b):
        locs = []
        while len(b) > 0:
          pos = self._f.tell()
          remaining = BLOCK_SIZE - pos
          # if the current file is full, close and open a new one.
          # todo: blob GCP
          if remaining == 0:
            self._f.close()
            self._f = next(self._file_gen)
            pos, remaining = 0, BLOCK_SIZE
          self._f.write(b[:remaining])
          locs.append((self._f.name, pos))
          b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def write_index(self, base_dir, name):
        self._write_globals(base_dir, name)

    # todo: write_index  and _write_globals GCP
    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
          pickle.dump(self, f)


class MultiFileReader:
  """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
  def __init__(self):
    self._open_files = {}
    # todo: client GCP

  def read(self, locs, n_bytes):
    b = []
    # todo: blob GCP
    for f_name, offset in locs:
      if f_name not in self._open_files:
        self._open_files[f_name] = open(f_name, 'rb')
      f = self._open_files[f_name]
      f.seek(offset)
      n_read = min(n_bytes, BLOCK_SIZE - offset)
      b.append(f.read(n_read))
      n_bytes -= n_read
    return b''.join(b)
  
  def close(self):
    for f in self._open_files.values():
      f.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
    return False

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
  def __init__(self, docs={}):
    """ Initializes the inverted index and add documents to it (if provided).
    Parameters:
    -----------
      docs: dict mapping doc_id to list of tokens
    """
    # stores document frequency per term
    self.df = Counter()
    # stores total frequency per term
    self.term_total = Counter()
    # stores posting list per term while building the index (internally), 
    # otherwise too big to store in memory.
    self._posting_list = defaultdict(list)
    # mapping a term to posting file locations, which is a list of 
    # (file_name, offset) pairs. Since posting lists are big we are going to
    # write them to disk and just save their location in this list. We are 
    # using the MultiFileWriter helper class to write fixed-size files and store
    # for each term/posting list its list of locations. The offset represents 
    # the number of bytes from the beginning of the file where the posting list
    # starts. 
    self.posting_locs = defaultdict(list)

    self.doc_id_title = defaultdict(list)

    self.doclen = defaultdict(list)

    for doc_id, tokens in docs.items():
      self.add_doc(doc_id, tokens)

  def add_doc(self, doc_id, tokens):
    """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).
    """
    w2cnt = Counter(tokens)
    self.term_total.update(w2cnt)
    # todo: add max value
    for w, cnt in w2cnt.items():
      self.df[w] = self.df.get(w, 0) + 1
      self._posting_list[w].append((doc_id, cnt))

  def write_index(self, base_dir, name):
    """ Write the in-memory index to disk. Results in the file: 
        (1) `name`.pkl containing the global term stats (e.g. df).
    """
    self._write_globals(base_dir, name)

  def write(self, base_dir, name):
    """ Write the in-memory index to disk and populate the `posting_locs`
  variables with information about file location and offset of posting
  lists. Results in at least two files:
  (1) posting files `name`XXX.bin containing the posting lists.
  (2) `name`.pkl containing the global term stats (e.g. df).
"""
    #### POSTINGS ####
    self.posting_locs = defaultdict(list)
    with closing(MultiFileWriter(base_dir, name)) as writer:
      # iterate over posting lists in lexicographic order
      for w in sorted(self._posting_list.keys()):
        self._write_a_posting_list(w, writer, sort=True)
    #### GLOBAL DICTIONARIES ####
    self._write_globals(base_dir, name)

  def _write_globals(self, base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
      pickle.dump(self, f)

  def __getstate__(self):
    """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
    state = self.__dict__.copy()
    del state['_posting_list']
    return state

  def posting_lists_iter(self, query):
    """ A generator that reads one posting list from disk and yields 
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
      for w in query:
        posting_list = []
        # read a certain number of bytes into variable b
        if w in self.posting_locs:
          locs = self.posting_locs[w]
          b = reader.read(locs, self.df[w] * TUPLE_SIZE)
          # convert the bytes read into 'b' to a proper posting list
          for i in range(self.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE: i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i+1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))

        yield w, posting_list


  def _write_a_posting_list(self, w, writer, sort=False):
    # taken from hw4 section of inverted_index
    # sort the posting list by doc_id
    pl = self._posting_list[w]
    if sort:
      pl = sorted(pl, key=itemgetter(0))
    # convert to bytes
    b = b''.join([(int(doc_id) << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                  for doc_id, tf in pl])
    # write to file(s)
    locs = writer.write(b)
    # save file locations to index
    self.posting_locs[w].extend(locs)


  @staticmethod
  def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
      return pickle.load(f)

  @staticmethod
  def delete_index(base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()


  @staticmethod
  def write_a_posting_list(b_w_pl):
    # taken from hw3 , inverted_index_colab.py
    ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...]) 
    and writes it out to disk as files named {bucket_id}_XXX.bin under the 
    current directory. Returns a posting locations dictionary that maps each 
    word to the list of files and offsets that contain its posting list.
    Parameters:
    -----------
      b_w_pl: tuple
        Containing a bucket id and all (word, posting list) pairs in that bucket
        (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
    Return:
      posting_locs: dict
        Posting locations for each of the words written out in this bucket.
    '''
    posting_locs = defaultdict(list)
    bucket, list_w_pl = b_w_pl
    # todo: add bucket name GCP
    with closing(MultiFileWriter('.', bucket)) as writer:
      for w, pl in list_w_pl: 
        # convert to bytes
        b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
      # save file locations to index
        posting_locs[w].extend(locs)
    return posting_locs