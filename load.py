from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("EPFL-ECEO/coralscapes")
ds.save_to_disk("coralscapes")