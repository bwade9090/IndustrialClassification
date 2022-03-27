import pandas as pd
from typing import List, Union
from gensim.corpora import Dictionary

class IndustrialTokenizer:
  def __init__(self,
               pre_tokens_fname='data/pre_toks_220323_2.csv',
               token_dictionary_fname='data/token_dictionary_220326.dic'):
    self.changer = {'a/s':' 수리 ', '써비스':'서비스', '짜장면':'자장면', '제단':'재단', '맛사지':'마사지', '마시지': '마사지', '프라스틱':'플라스틱', '돌보미':'돌봄'}
    self.separators = ['+', '-', '[', ']', '>', '<', '/', 'ㆍ',':', ';', '?', '？', '&', ',', '，', '.', '．', '(', ')', '및']
    self.pre_tokens = self.get_pre_tokens(pre_tokens_fname)
    self.dictionary = Dictionary.load(token_dictionary_fname)

  def get_pre_tokens(self, fname: str) -> List[str]:
    return sorted(list(pd.read_csv(fname)['0']),key=len, reverse=True)

  def tokenize_and_encode(self, s: Union[str, float]) -> List[int]:
    return self.encode(self.tokenize(s))

  def encode(self, tokens: List[str]) -> List[int]:
    return list(map(lambda i: i + 2, self.dictionary.doc2idx(tokens)))

  def tokenize(self, s: Union[str, float]) -> List[str]:
    return self.replace(s).split() if pd.notna(s) else []

  def replace(self, s: str) -> str:
    return self.extract_pre_tokens(self.separate(self.change(s)))

  def extract_pre_tokens(self, s: str) -> str:
    for token in self.pre_tokens:
      s = s.replace(token, f' {token} ')
    return s

  def separate(self, s: str) -> str:
    for seq in self.separators:
      s = s.replace(seq, ' ')
    return s

  def change(self, s: str) -> str:
    for asis, tobe in self.changer.items():
      s = s.replace(asis, tobe)
    return s
