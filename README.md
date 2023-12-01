# IR - final project report

## Part I: Biomedical dataset for STAR dense retrieval
The implementation source code is based on the following projects
1. DRhard -- https://github.com/jingtaozhan/DRhard 
2. JTR -- https://github.com/CSHaitao/JTR/tree/main

Run the following codes for biomedical dataset preprocess <br>
`python preprocess_bio.py --data_type 0`

STAR: use the provided STAR model to compute query/passage embeddings and perform similarity search on the biomedical dataset. <br>
`python inference.py --data_type doc --max_doc_length 512 --mode bio-train`
 
Tree Initialization 
After embedding documents and query, we can initialize the tree using recursive k-means.<br>
Run the following codes in JTR repo:<br>
`python construct_tree.py`
