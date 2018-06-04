# [Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/)

# Requirements
* Python 3
* Pytorch 0.4.0

# Installation


* Clone this repo.
```
git clone --recurse-submodules https://github.com/jabalazs/implicit_emotion.git
cd implicit_emotion
```

**We recommend doing all the steps below within a conda environment**

* Make sure you already have python 3 and pytorch installed

* Install the other dependencies
```
pip install -r requirements.txt
```

* Install Matplotlib and Scikit learn
```
conda install matplotlib scikit-learn
```

* Get the data by running the following command and typing your password when prompted
```
./get_data.sh <USERNAME>
```

* Run the preprocessing script
```
./preprocess.sh
```

To test if you installed everything correctly run `python run.py --help`.
