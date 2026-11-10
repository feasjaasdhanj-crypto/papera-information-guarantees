"""R540 preparation of user-returned feature ZIP (research code, Python).
Usage: python R540_feature_assembly.py /path/to/RETURN_TO_CHATGPT_EEG_R539.zip /path/to/EEG_features_extracted.pkl
The EEG features are .csv.gz members inside an outer ZIP; no network operations.
"""
import zipfile,gzip,io,sys,hashlib,pandas as pd
from pathlib import Path
p=Path(sys.argv[1]);out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(p) as z:
    z.testzip()
    names=[x for x in z.namelist() if x.endswith('.csv.gz')]
    x=pd.concat([pd.read_csv(io.BytesIO(gzip.decompress(z.read(k)))) for k in names],ignore_index=True)
    assert x.subject.nunique()==106 and len(x)==9536 and x.shape[1]==198
x.to_pickle(out)
print('PASS',len(x),x.subject.nunique(),'SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
