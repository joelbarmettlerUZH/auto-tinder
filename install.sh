# python3 -m venv env
python3 -m virtualenv env
source env/bin/activate

python3 -m pip  install --upgrade pip
python3 -m pip install -r requirements.txt
deactivate
