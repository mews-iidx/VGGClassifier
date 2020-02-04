# VGGClassifier


## clone 

```bash
git clone https://github.com/mews-iidx/VGGClassifier
```

## model deployment

```bash
mv VGGClassifier /data/models/
```

next, You deploy the model on Web GUI.

## predict from API

```bash
cd VGGClassifier
pip3 install -r  requirements.txt

# predict from API
python3 request.py imgs/cat9302332_TP_V.jpg http://XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# standalone test without Grace
python3 run_single.py 
```
