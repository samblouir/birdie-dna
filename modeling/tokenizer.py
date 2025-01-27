# modeling/tokenizer.py
"""
=============================================================================
DNA TOKENIZER (ByT5-based)
-------------
Uses Hugging Face's AutoTokenizer to load ByT5, which tokenizes text by
converting them to their UTF-8 byte index representations (or longer, depending on the character/language).
=============================================================================
"""
import config
from transformers import AutoTokenizer
from typing import List, Union
import numpy as np

class DNATokenizer:
	"""
	A tokenizer wrapper around ByT5's byte-level tokenizer.

	Attributes:
		tokenizer: The loaded Hugging Face tokenizer (ByT5).
		vocab_size (int): Size of the ByT5 vocabulary.
	"""
	def __init__(self, config=None):
		"""
		Args:
			config (dict): Doesn't do anything at the moment.
		"""

		if config is None:
			config = {}
			
		# Load the ByT5 tokenizer via HuggingFace's AutoTokenizer
		self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", use_fast=False)

		# 0 is a common token for pad tokens
		self.pad_token_id = self.tokenizer.pad_token_id
		if self.pad_token_id is None:
			self.tokenizer.pad_token_id = 0
			self.pad_token_id = 0

		self.vocab_size = config.get("vocab_size", self.tokenizer.vocab_size)

	def encode(self, text: str):
		"""
		Tokenizes the input text at the byte level using ByT5's tokenizer.
		Adds BOS and EOS tokens.
		
		Args:
			text (str): DNA (or any) string to encode.
		
		Returns:
			List[int]: List token IDs.
		"""
		try:
			return np.int32(self.tokenizer.encode(text, add_special_tokens=True))
		except:
			return [np.int32(self.tokenizer.encode(t)) for t in text]

	def decode(self, token_ids: Union[List[int], List[List[int]], np.ndarray]) -> Union[str, List[str]]:
		"""
		Converts token IDs back into a string using ByT5's decoding.
		Handles single lists of IDs or batches (list of lists).
		"""
		try:
			return self.tokenizer.decode(token_ids, skip_special_tokens=True)
		except:
			return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
	

if __name__ == "__main__":

	# Quick test
	tokenizer = DNATokenizer()
	text = "ATGC"
	input_ids = tokenizer.encode(text)
	decoded_input_ids = tokenizer.decode(input_ids)
	print(f"*" * 60,)
	print(f"  text: {text}")
	print(f"  input_ids: {input_ids}")
	print(f"  decoded_input_ids: {decoded_input_ids}")
