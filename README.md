## Overview 
This repo includes the code for the paper "OTTAWA: Optimal Transport Adaptive Word Aligner for Hallucination and Omission Translation Errors Detection" (Findings of ACL 2024).
## Setup

```bash
TMP_FILE_PATH="" # The path to the temporary file, including embeddings 
python aligner.py  --mode dump_emb # Prepare the embeddings

$TASK="hall"  # hall or omi
python aligner.py  --mode run_hall  --error_type $TASK
```

## Code structure

* `aligner.py`: The main code for the alignment
* `utils/hall_methods.py` includes the implementation of our method, in particualr, the `ot_align_fwd_rev` function
    
## Reference
```
@inproceedings{huang-etal-2024-ottawa,
    title = "{OTTAWA}: Optimal {T}ranspor{T} Adaptive Word Aligner for Hallucination and Omission Translation Errors Detection",
    author = "Huang, Chenyang  and
      Ghaddar, Abbas  and
      Kobyzev, Ivan  and
      Rezagholizadeh, Mehdi  and
      Zaiane, Osmar  and
      Chen, Boxing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.377",
    doi = "10.18653/v1/2024.findings-acl.377",
    pages = "6322--6334",
    
}
```

