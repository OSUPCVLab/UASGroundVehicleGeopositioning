# VehicleGeopositioning

## Set up
### Copy this Repo and SuperGlue's repo
Run the following commands to clone this repo and SuperGlue's repo:

TODO: git setup

`$ git clone https://github.com/OSUPCVLab/VehicleGeopositioning.git`

`$ cd VehicleGeopositioning`

`$ git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git`

### Setup Dependancies
Run the following commands to set up the dependancies:

`$ pip3 install -r requirements.txt`

### Google API
1. Get a google maps API key. Use the following link to set up the api key. Make sure you copy it (do not share API keys, each person should use their own)
[https://developers.google.com/maps/documentation/embed/get-api-key](https://developers.google.com/maps/documentation/embed/get-api-key)
TODO: more directions here when navigating the google API
2. Open the `keys.py` document and paste in your API key into the `GOOGLE_API_KEY` field 

## Running the code
1. `main.py` there is the script which runs everything. To run in simpily type `python3 main.py` into the command line.
2. Here is how to use custom data. There are two predefined arguments `--framesDir` and `--dataDir`. Currently there are defined to point to the sample data. The convention is that if an image `data/images/img.png` exists, then there is a corresponding `data/parameters/img.txt` that supplies meta data for the image. Each image's meta data should provide the same information in the same format as currently exists in the sample data. 
ex. `GPS = 40.01265351729508,-83.00969233386985
height = 70
rotation = 0`
