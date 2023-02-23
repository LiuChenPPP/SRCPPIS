# SRCPPIS
The source code of "SRCPPIS: Utilizing structure based masking
mechanism and residued based graph neural network
with cross attention to predict protein protein
interaction site"

# Dataset
Since our dataset exceeds the maximum upload capacity of Github, please kindly download the data in "https://drive.google.com/drive/folders/1S1G_wf40CJZGtnIr3-2gYLiztKxl_wqV?usp=sharing".




If you want to test the performance on your own data, please make sure you install the following software and download the corresponding databases:

(1) PSSM(L*20) can be obtained by BLAST("https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/") and UNIREF("https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz").

(2) HHM(L*20) can be obtained by HH-suite("https://github.com/soedinglab/hh-suite") and Uniclust("https://gwdu111.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite.tar.gz").

(3)Protbert(L*1024) can be obtained by "https://huggingface.co/Rostlab/prot_bert".

# Reproduction
python3 train.py
# Test
python3 test.py


