# Leukemia classification using googlenet architecture

## Streamlit demo
![](streamlit/demo.gif)

## Requirements

- python runtime >=3.10 <=3.12
- tensorflow
- streamlit
- opencv
- Etc, see ```streamlit/requirements.txt``` for details.

## How to run streamlit

First, you have to clone this repository
```
git clone https://github.com/Egi-putrA/googlenet-leukemia-classification.git
```
Then, open streamlit subdirectory
```
cd googlenet-leukemia-classification/streamlit
```
### Optional: use python virtual environments before install dependecies
Create virtual env
```
python -m venv venv
```
Activate virtual env
```
venv\Scripts\activate.bat # for windows
venv\bin\activate # for linux
```

### Install dependencies
install dependencies from requirements.txt
```
pip install -r requirements.txt
```
### Download trained model
You can download my trained model in [here](https://github.com/Egi-putrA/googlenet-leukemia-classification/releases/).
Make sure to change model path in main.py
```python
...
@st.cache_resource
def load_model():
    return tf.saved_model.load('path/to/model').signatures['serving_default']
...
```
### Run streamlit app
```
streamlit run main.py
```