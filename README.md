# amanita


To create and activate the venv and install dependencies:
```bash
$ python -m venv .amanita
$ source .amanita/bin/activate
$ pip install -r requirements.txt
```

To install wget for dataset download:
```bash
$ sudo apt-get update
$ sudo apt-install wget
```

To install the dataset tools:
```bash
$ cd; git clone https://github.com/BohemianVRA/FungiTastic.git
$ cd FungiTastic/dataset/
$ python download.py --metadata --images --subset "m" --size "300" --save_path "./" 
```