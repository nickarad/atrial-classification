1. Download Dataset
    $ wget -r -np http://www.physionet.org/physiobank/database/afpdb/

2. Copy paste afpdb folder here 
    $ sudo mv www.physionet.org/physiobank/database/afpdb .

3. Create CSV files 
    $ python dat2csv.py

4. Create training and test dataset
    $ mkdir training test

5. Split manually dataset (afpdbCSV) to training and test:
    * My dataset has 180 training samples (90 normal + 90 abnormal) and   20 testing samples (10 normal + 10 abnormal)