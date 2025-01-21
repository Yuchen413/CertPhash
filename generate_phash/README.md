# Existing Phash System

## pyPhotoDNA
We follow the: [repo](https://github.com/jankais3r/pyPhotoDNA)

### Setup
1)  Run `install.bat` if you are on Windows, or `install.sh` if you are on a Mac or Linux. Then you should get a file with suffix '.dll'
2)  Once the setup is complete, run `WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py` to generate hashes.


If you want to learn more about PhotoDNA, head over to [jPhotoDNA](https://github.com/jankais3r/jPhotoDNA).


## PDQ
We follow the: [repo](https://github.com/faustomorales/pdqhash-python)

### Setup
1) `pip insatll pdqhash`
2) `python get_pdq_hash.py`

## NeuralHash

We follow the [repo](https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing) to extract the model.
Hash generation is under ./attack folder, building upn this [repo](https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing).