cd ~
git clone https://github.com/jimchen2/indoor_localization_oxford_dataset
python -m venv env
source env/bin/activate
cd indoor_localization_oxford_dataset
pip install -r requirements.txt
sudo apt install unzip


rm -r data
wget https://jimchen4214-public.s3.us-east-1.amazonaws.com/other/modified_data.zip
mv modified_data.zip data.zip
unzip data.zip