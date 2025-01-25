# data_utils/dataloader.py
"""
==============================================================================
ALL-IN-ONE DATA LOADER
-------------------------------------
TODO: Needs to be split into several parts...

Handles:
  1) Downloading & extracting 
  2) Parsing FASTA files (using BioPython).
  3) Splitting into train/validation/test sets.
  4) Creating data batches (with 'input_ids', 'label_ids', and 'segment_ids').
  

Contents:
  - DatasetEntry, DatasetWrapper: Classes to hold dataset items & structures
  - download_datasets(): Download & optionally extract compressed archives
  - load_dataset(): High-level function to get a dictionary of splits
  - create_clm_batch_generator(): Yields CLM/next token prediction training batches with 
								  (input_ids, label_ids, segment_ids)
==============================================================================
"""

import os
import urllib.request
import zipfile
import tarfile
import math
import typing
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import multiprocessing as mp
import utils  # For debug_print, etc.

##############################################################################
# 1) DatasetEntry and DatasetWrapper
##############################################################################

class DatasetEntry:
	"""
	Represents a single textual/FASTA entry (e.g., one DNA sequence).
	
	Attributes:
		text (str): The raw text of this entry (e.g., 'ACGT...').
	"""
	def __init__(self, text: str):
		# Store the text string
		self.text = text

	def __getitem__(self, key: str):
		"""
		Allows dictionary-like access for convenience. E.g., entry["text"] -> "ACGT..."
		"""
		return self.__dict__[key]

	def __str__(self):
		"""
		Returns a short string with the length of the text.
		"""
		return f"DatasetEntry(len(text)={len(self.text)})"


class DatasetWrapper:
	"""
	A container holding either:
	  - A single list of DatasetEntry objects, OR
	  - A dictionary of splits (like {'train': {...}, 'validation': {...}}).

	Attributes:
		data (dict or list): The underlying data structure.
		name (str): An optional name for this dataset or split.
		tokenizer (Any): Potentially store a tokenizer for easy reference.
		metadata (dict): Additional info stored as keyword args.
	"""
	def __init__(self, data, name=None, tokenizer=None, **kwargs):
		self.data = data
		self.name = name
		self.tokenizer = tokenizer
		self.metadata = kwargs

	def __len__(self):
		"""
		If 'data' is a dict with sub-dicts, sums the length of each sub-list.
		Otherwise, returns the length of the list directly.
		"""
		if isinstance(self.data, dict):
			total = 0
			for k in self.data:
				if "entries" in self.data[k]:
					total += len(self.data[k]["entries"])
			return total
		else:
			return len(self.data)

	def __getitem__(self, idx):
		"""
		If 'data' is a dict, flattens all 'entries' into one list for direct indexing.
		Otherwise, returns the item at idx
		"""
		if isinstance(self.data, dict):
			flattened = []
			for key in self.data:
				if "entries" in self.data[key]:
					flattened.extend(self.data[key]["entries"])
			return flattened[idx]
		else:
			return self.data[idx]

	def __iter__(self):
		"""
		Allows iteration over all entries: yields from each sub-list in a dict,
		or simply from the single list.
		"""
		if isinstance(self.data, dict):
			for key in self.data:
				if "entries" in self.data[key]:
					yield from self.data[key]["entries"]
		else:
			yield from self.data

	def __str__(self):
		"""
		Returns a user-friendly description (name + total number of entries).
		"""
		total_len = len(self)
		return f"DatasetWrapper('{self.name}' with {total_len} entries)"


##############################################################################
# 2) Downloading and Extracting Archives
##############################################################################

def get_absolute_path(dir_name: str = "dataset", filename: str = "") -> str:
	"""
	Constructs an absolute path for a directory + optional filename.

	Args:
		dir_name (str): Base directory name.
		filename (str): Optional filename to append.

	Returns:
		str: The combined absolute path.
	"""
	base_dir = os.path.abspath(dir_name)
	return os.path.join(base_dir, filename)

def download_file(download_dir_name: str, filename: str, url: str) -> str:
	"""
	Downloads a file from a given URL into a specified directory (download_dir_name) 
	with the given filename. Skips downloading if the file already exists.

	Returns the absolute local path of the downloaded file.
	"""
	destination = get_absolute_path(download_dir_name, filename)

	# If file exists, skip download
	if os.path.exists(destination):
		utils.debug_print(f"File '{destination}' already exists, skipping download.")
		return destination

	# Make sure the directory exists
	os.makedirs(os.path.dirname(destination), exist_ok=True)

	# We'll track progress with a TQDM bar
	progress_bar = None

	def progress_hook(block_num, block_size, total_size):
		nonlocal progress_bar
		if progress_bar is None:
			progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
		progress_bar.update(block_size)

	# Debug info & actual download
	utils.debug_print(f"Downloading from URL: {url} -> {destination}")
	urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
	if progress_bar:
		progress_bar.close()

	return destination

def detect_if_compressed_dir(filepath: str) -> dict:
	"""
	Checks if 'filepath' is a recognized compressed archive (zip, tar, tar.gz).
	Returns a dictionary describing whether it is recognized and how to unpack.

	Returns:
		dict containing:
		  "is_compressed_dir": bool
		  "filetype": str (e.g., "zip", "tar", "tar.gz", or other extension)
		  "unpack_dir": str or None (suggested extraction directory)
	"""
	# Check .zip
	if zipfile.is_zipfile(filepath):
		return {
			"is_compressed_dir": True,
			"filetype": "zip",
			"unpack_dir": filepath.rsplit(".", 1)[0],
		}

	# Check tar.gz
	try:
		with tarfile.open(filepath, mode="r:gz"):
			return {
				"is_compressed_dir": True,
				"filetype": "tar.gz",
				"unpack_dir": filepath.rsplit(".", 2)[0],
			}
	except tarfile.ReadError:
		pass

	# Check tar
	try:
		with tarfile.open(filepath, mode="r:"):
			return {
				"is_compressed_dir": True,
				"filetype": "tar",
				"unpack_dir": filepath.rsplit(".", 1)[0],
			}
	except tarfile.ReadError:
		pass

	# Otherwise, not recognized
	extension = filepath.rsplit(".", 1)[-1]
	return {
		"is_compressed_dir": False,
		"filetype": extension,
		"unpack_dir": None,
	}

def unpack_compressed_dir(filepath: str) -> None:
	"""
	If 'filepath' is a recognized compressed file, extracts it into a
	subdirectory (unless already extracted).
	"""
	info = detect_if_compressed_dir(filepath)
	if not info["is_compressed_dir"]:
		return  # Not compressed in a recognized format

	extraction_dir = info["unpack_dir"]
	if not extraction_dir:
		return

	# If extraction_dir exists, skip
	if os.path.exists(extraction_dir):
		utils.debug_print(f"Extraction directory '{extraction_dir}' already exists, skipping.")
		return

	# Extract
	os.makedirs(extraction_dir, exist_ok=True)
	utils.debug_print(f"Extracting '{filepath}' to '{extraction_dir}'")

	if info["filetype"] == "zip":
		with zipfile.ZipFile(filepath, "r") as zf:
			zf.extractall(extraction_dir)
	elif info["filetype"] == "tar.gz":
		with tarfile.open(filepath, "r:gz") as tar:
			tar.extractall(extraction_dir)
	elif info["filetype"] == "tar":
		with tarfile.open(filepath, "r:") as tar:
			tar.extractall(extraction_dir)
	else:
		raise ValueError(f"Unrecognized compressed file: {filepath}")

def download_datasets(files_to_download: list) -> dict:
	"""
	Given a list of file-descriptors (dicts) specifying how to download/unpack,
	downloads each file if needed and unpacks it if recognized as compressed.

	Expects each dict to have:
	  - 'download_dir_name'
	  - 'filename'
	  - 'url'
	  - 'files_to_extract' (optional list of expected files)

	Returns a dict with 'success' and 'failure' lists of filenames.
	"""
	results = {"success": [], "failure": []}
	for fdesc in files_to_download:
		try:
			local_path = download_file(
				download_dir_name=fdesc["download_dir_name"],
				filename=fdesc["filename"],
				url=fdesc["url"]
			)
			unpack_compressed_dir(local_path)
			results["success"].append(fdesc["filename"])
		except Exception as e:
			utils.debug_print(
				f"ERROR downloading/unpacking '{fdesc['filename']}': {str(e)}"
			)
			results["failure"].append(fdesc["filename"])
	return results


##############################################################################
# 3) FASTA Parsing & Dataset Splitting
##############################################################################

def open_fasta(fasta_path: str, print_every: int = 2048) -> list:
	"""
	Parses a FASTA file at 'fasta_path', reading each sequence record into
	a DatasetEntry. Uses BioPython's SeqIO.

	Args:
		fasta_path (str): Path to the FASTA file.
		print_every (int): How often to print debug info (every N records).

	Returns:
		list of DatasetEntry objects.
	"""
	utils.debug_print(f"Parsing FASTA: {fasta_path}")
	entries = []
	count = 0

	with open(fasta_path, "r") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			seq_str = str(record.seq)
			entries.append(DatasetEntry(text=seq_str))
			count += 1
			if (0 < print_every) and (count % print_every) == 0:
				utils.debug_print(f"  Parsed {count} records from {fasta_path}")

	utils.debug_print(f"Finished parsing {count} records from {fasta_path}")
	return entries

def load_dataset(
	config: dict = None,
	validation_split_percentage: float = 0.05,
	test_split_percentage: float = 0.05,
	dataset_rng_seed: typing.Union[int, typing.Iterable[int]] = 1337,
	files_to_download: list = None,
	download_dir_name: str = "dataset",
	dataset_name: str = None,
	**kwargs,
) -> dict:
	"""
	High-level function to:
	  1) Download and unpack files (if needed).
	  2) Parse the relevant FASTA files.
	  3) Split data into train/validation/test sets.
	  4) Return a dict with keys "train"/"validation"/"test" of DatasetWrapper objects.

	Args:
		config (dict): Configuration dictionary (can be unused, but included for convenience).
		validation_split_percentage (float): Fraction of data for validation.
		test_split_percentage (float): Fraction of data for test.
		dataset_rng_seed (int or Iterable[int]): Random seed or seeds (if multiple files).
		files_to_download (list): List of dict instructions for file download.
		download_dir_name (str): Base directory for downloads.
		dataset_name (str): Optional name to tag the dataset.
		**kwargs: Additional arguments for future expansion.

	Returns:
		dict: {"train": DatasetWrapper, "validation": DatasetWrapper, "test": DatasetWrapper}
	"""
	if config is None:
		config = {}
	# If no specific file instructions given, default to a sample T2T genome
	if files_to_download is None:
		files_to_download = [
			{
				"download_dir_name": download_dir_name,
				"filename": "T2T.zip",
				"url": (
					"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
					"GCF_009914755.1/download?include_annotation_type=GENOME_FASTA&"
					"include_annotation_type=GENOME_GFF&include_annotation_type=RNA_FASTA&"
					"include_annotation_type=CDS_FASTA&include_annotation_type=PROT_FASTA&"
					"include_annotation_type=SEQUENCE_REPORT&hydrated=FULLY_HYDRATED"
				),
				"files_to_extract": ["GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"]
			},
		]

	# Step 1: Download & extract
	download_results = download_datasets(files_to_download)
	utils.debug_print(f"Download results: {download_results}")

	# Prepare random generator
	if isinstance(dataset_rng_seed, int):
		rng = np.random.default_rng(dataset_rng_seed)

	# We'll accumulate each subset in these dicts
	all_train_entries = {}
	all_val_entries = {}
	all_test_entries = {}

	# Minimum needed if we have train, val, test splits
	min_needed = 1 + int(validation_split_percentage > 0) + int(test_split_percentage > 0)

	# Step 2: Parse each specified file, then split
	for idx, item in enumerate(files_to_download):
		fname = item["filename"]
		# If multiple seeds are provided, pick the i-th for the i-th file
		if isinstance(dataset_rng_seed, typing.Iterable):
			seeds_list = list(dataset_rng_seed)
			chosen_seed = seeds_list[idx] if idx < len(seeds_list) else 1337
			rng = np.random.default_rng(chosen_seed)

		# We assume extracted contents go in "filename-without-extension"
		extract_subdir = fname.rsplit(".", 1)[0]
		extracted_dir_path = get_absolute_path(item["download_dir_name"], extract_subdir)

		# For each file we expect after extraction
		for expected_file in item.get("files_to_extract", []):
			actual_fasta_path = None
			# Recursively find the expected_file
			for root, dirs, files in os.walk(extracted_dir_path):
				if expected_file in files:
					actual_fasta_path = os.path.join(root, expected_file)
					break

			if not actual_fasta_path or not os.path.exists(actual_fasta_path):
				raise FileNotFoundError(f"Could not find '{expected_file}' in '{extracted_dir_path}'.")

			# Parse the FASTA
			entries = open_fasta(actual_fasta_path, print_every=50_000)
			indices = np.arange(len(entries))

			# Check we have enough data to split
			if len(indices) < min_needed:
				raise ValueError(
					f"File '{actual_fasta_path}' has only {len(indices)} entries; cannot split."
				)

			rng.shuffle(indices)
			n = len(indices)
			n_val = int(n * validation_split_percentage)
			n_test = int(n * test_split_percentage)
			n_train = n - n_val - n_test

			train_idx = indices[:n_train]
			val_idx   = indices[n_train : n_train + n_val]
			test_idx  = indices[n_train + n_val :]

			# Convert to the dictionary format
			tr_entries = [
				dict(entry=entries[i], source=fname, split="train") for i in train_idx
			]
			va_entries = [
				dict(entry=entries[i], source=fname, split="validation") for i in val_idx
			]
			te_entries = [
				dict(entry=entries[i], source=fname, split="test") for i in test_idx
			]

			# Store in dictionaries keyed by filename
			all_train_entries[fname] = {"entries": tr_entries, "source": fname, "split": "train"}
			all_val_entries[fname]   = {"entries": va_entries, "source": fname, "split": "validation"}
			all_test_entries[fname]  = {"entries": te_entries, "source": fname, "split": "test"}

	utils.debug_print(f"Loaded {len(all_train_entries)} files with splits")
	# Wrap in DatasetWrappers
	ds_train = DatasetWrapper(all_train_entries, name="train", dataset_name=dataset_name)
	ds_val   = DatasetWrapper(all_val_entries,   name="validation", dataset_name=dataset_name)
	ds_test  = DatasetWrapper(all_test_entries,  name="test", dataset_name=dataset_name)

	# Safety check
	if (len(ds_train) + len(ds_val) + len(ds_test)) == 0:
		raise RuntimeError("No data loaded. Check your config or dataset paths.")

	return {
		"train": ds_train,
		"validation": ds_val,
		"test": ds_test,
	}

##############################################################################
# 4) BATCH GENERATOR FOR CAUSAL LANGUAGE MODELING
##############################################################################

def _batcher(
	dataset: typing.Union[typing.List[typing.Any], 'DatasetWrapper'],
	tokenizer,
	batch_size: int = 8,
	max_sequence_length: int = 128,
	minimum_sample_length = 32,
	num_workers: int = 1,
	pad_token_id: typing.Optional[int] = None,
	worker_idx: int = 0,
	infinite_loop: bool = None,
	output_queue=None,
	**kwargs
	):

	def data_iter(ds):
		"""If infinite_loop is True, yield items repeatedly."""
		if infinite_loop:
			while True:
				yield from ds
		else:
			yield from ds

	batch_input_ids = []
	batch_label_ids = []
	batch_segment_ids = []


	for item in data_iter(dataset):
		# Extract raw text
		if isinstance(item, dict) and ("entry" in item) and hasattr(item["entry"], "text"):
			text = item["entry"].text
		else:
			text = item  # Assume item is directly a string or something tokenize-able

		# all_chunks = np.array_split(text, num_workers)
		# our_chunk_text = all_chunks[worker_idx] if worker_idx < len(all_chunks) else []
		chunk_size = (len(text) + num_workers - 1) // num_workers
		start = worker_idx * chunk_size
		end = start + chunk_size
		our_chunk_text = text[start:end]


		if not len(our_chunk_text):
			continue

		# Tokenize the text
		our_chunk = tokenizer.encode(our_chunk_text)

		# Split token_ids across workers (if using multiple workers)

		# Now process in windows of exactly (max_sequence_length + 1)
		# so input_ids has length max_sequence_length and label_ids has length max_sequence_length.
		for start in range(0, len(our_chunk), max_sequence_length):
			end = start + max_sequence_length + 1
			window = our_chunk[start:end]

			# If the window is too short, discard it

			if len(window) < minimum_sample_length:
				continue
			
			# Slightly non-optimal right now.
			# Placeholder to add sequence packing.
			segment_id = (len(batch_input_ids) + 1)
			segment_ids = [segment_id] * len(window) + [0]
			# The first max_sequence_length tokens are input, shifted by 1 for labels
			input_ids = window[:-1]
			label_ids = window[1:]

			batch_input_ids.append(input_ids)
			batch_label_ids.append(label_ids)
			batch_segment_ids.append(segment_ids)

			# If we have enough samples to create a batch, yield a batch
			if len(batch_input_ids) == batch_size:
				prepared_batch = {
					"input_ids":   np.array(batch_input_ids,  dtype=np.int64),
					"label_ids":   np.array(batch_label_ids,  dtype=np.int64),
					"segment_ids": np.array(batch_segment_ids, dtype=np.int64),
				}
				if output_queue is not None:
					while True:
						# utils.debug_print(f"  worker_idx: {worker_idx:,}. Putting batch into the queue (size: {output_queue.qsize()})")
						try:
							output_queue.put(prepared_batch)
						except Exception as e:
							time.sleep(0.1)
				batch_input_ids.clear()
				batch_label_ids.clear()
				batch_segment_ids.clear()

# This is a wrapper function to allow passing kwargs to the batcher
def batcher(kwargs):
	return _batcher(**kwargs)


def create_clm_batch_generator(
	dataset: typing.Union[typing.List[typing.Any], 'DatasetWrapper'],
	tokenizer,
	max_sequence_length: int = 128,
	batch_size: int = 8,
	pad_token_id: typing.Optional[int] = None,
	worker_idx: int = 0,
	num_workers: int = 1,
	process_idx: int = 0,
	num_processes: int = 1,
	two_dimensional_split: bool = True,
	infinite_loop: bool = None,
	split: str = None,
	ds_shuiffle_rng_seed: int = 1337,
	ds_shuffle_np_rng: np.random.Generator = None,
	minimum_sample_length = 32,
	ignore_label_id: int = -100,
	number_of_batches: int = None,
	config = None,
	**kwargs
):
	"""
	Yields batches suitable for causal language modeling. Each batch is a dict:
	  {
		"input_ids":   np.ndarray of shape (batch_size, max_sequence_length),
		"label_ids":   np.ndarray of shape (batch_size, max_sequence_length),
		"segment_ids": np.ndarray of shape (batch_size, max_sequence_length)
	  }

	This version uses Option A: non-overlapping windows of (max_sequence_length + 1).
	Each final example is exactly `max_sequence_length` tokens of input,
	shifted by 1 for the labels. No off-by-one mismatch with segment_ids.

	Args:
		dataset: A DatasetWrapper or list of items. Each item is either a dict
				 with item["entry"].text or a raw string.
		tokenizer: An object with an 'encode' method to convert text -> int tokens.
		max_sequence_length: The fixed length for the input_ids/label_ids dimension.
		batch_size: Number of sequences per yielded batch.
		pad_token_id: ID for padding (not used below if you discard short windows).
		ignore_label_id: Token ID to assign for ignoring certain positions.
		**kwargs: Extra arguments for future expansions.

	Yields:
		dict: A dictionary with "input_ids", "label_ids", and "segment_ids" arrays.
	"""

	assert 0 < batch_size
	assert 0 < max_sequence_length
	assert 0 < num_workers
	assert 0 <= worker_idx

	if config is None:
		config = {}

	# Decide if we loop forever for the train split
	if infinite_loop is None and split is not None:
		if "train" in split.lower():
			infinite_loop = True
		else:
			infinite_loop = False

	# Creates a new RNG for shuffling if not provided
	if ds_shuffle_np_rng is None:
		ds_shuffle_np_rng = np.random.default_rng(ds_shuiffle_rng_seed)

	# If user doesn't provide pad_token_id, attempt to fetch from tokenizer
	if pad_token_id is None:
		pad_token_id = getattr(tokenizer, "pad_token_id", None)
		if pad_token_id is None:
			raise ValueError("No pad_token_id provided and tokenizer has no pad_token_id.")


	output_queue = mp.Queue(maxsize=2 * num_workers)

	total_num_workers = (num_workers * num_processes)
	worker_id_offset = (process_idx * num_workers) + worker_idx

	for _worker_idx in range(num_workers):
		worker_idx = (_worker_idx + worker_id_offset)

		batcher_kwargs = dict(
			worker_idx=worker_idx,
			num_workers=total_num_workers,

			dataset=dataset,
			tokenizer=tokenizer,

			batch_size=batch_size,
			max_sequence_length=max_sequence_length,
			minimum_sample_length=minimum_sample_length,
			pad_token_id=pad_token_id,

			infinite_loop=infinite_loop,
			output_queue=output_queue,
		)

		mp.Process(target=batcher, args=(batcher_kwargs,)).start()
		# utils.debug_print(f"  Started worker {worker_idx:,} of {num_workers:,} (process {process_idx:,} of {num_processes:,}, total_num_workers: {total_num_workers:,})")
		# batcher(batcher_kwargs)

	def _batch_generator():
		
		while True:
			yield output_queue.get()

	batch_generator = _batch_generator()

	# If infinite, keep yielding forever
	if (number_of_batches is None) and (infinite_loop):
		return batch_generator

	
	deterministic_batcher_kwargs = {}
	for key, value in batcher_kwargs.items():
		# Skip items that might have different hashes each run
		if " object at 0x" in repr(value):
			continue
		deterministic_batcher_kwargs[key] = value
	deterministic_batcher_kwargs['num_workers_dataset'] = num_workers
	deterministic_batcher_kwargs['process_idx'] = process_idx
	deterministic_batcher_kwargs['num_processes'] = num_processes
	deterministic_batcher_kwargs['split'] = split
	deterministic_batcher_kwargs['number_of_batches'] = number_of_batches
	deterministic_batcher_kwargs['num_workers'] = num_workers
	
	deterministic_batcher_kwargs = utils.sort_dict(deterministic_batcher_kwargs)
	cache_key = utils.obj_to_str_hash(deterministic_batcher_kwargs)
	try:
		return utils.quick_load(cache_key)
	except Exception as e:
		utils.debug_print(f"  Creating {number_of_batches} batches for {split} (this will appear to hang, give it ~5 minutes)")
		# Otherwise, gather exactly 'number_of_batches' and return
		progress_bar = tqdm(total=number_of_batches, disable=(worker_idx != 0), desc=f"Preparing {number_of_batches} batches for {split}")
		batches = []
		for _ in range(number_of_batches):
			# utils.debug_print(f"  Creating valiation batch {_} of {number_of_batches}")
			progress_bar.update(1)
			batches.append(next(batch_generator))
		progress_bar.close()
		utils.quick_save(cache_key, batches)
		return batches


if __name__ == "__main__":
	dataset = load_dataset()
	ds_train = dataset["train"]

	from modeling.tokenizer import DNATokenizer
	tokenizer = DNATokenizer()



	
	













