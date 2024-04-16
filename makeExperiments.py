import itertools
import os

# features = ['--SuperGlue False --LoFTR True ']
features = ['--SuperGlue True  --LoFTR True ',
            '--SuperGlue True  --LoFTR False',
            '--SuperGlue False --LoFTR True ']
filters = ['--filterCars True  --filterRoads True  --filterBuildings True ',
           '--filterCars True  --filterRoads True  --filterBuildings False',
           '--filterCars True  --filterRoads False --filterBuildings True ',
           '--filterCars True  --filterRoads False --filterBuildings False',
           '--filterCars False --filterRoads True  --filterBuildings True ',
           '--filterCars False --filterRoads True  --filterBuildings False',
           '--filterCars False --filterRoads False --filterBuildings True ',
           '--filterCars False --filterRoads False --filterBuildings False']
transformations = ['--homography True ',
                   '--homography False']

mainList = [features, filters, transformations]

escapeStr = '\''

for l in list(itertools.product(*mainList)):
    os.system(
        f"python3 main.py {str(l).replace(',', '').replace(escapeStr, '').replace('(', '').replace(')', '')}")
    # print('python3 main.py',str(l).replace(',', '').replace('\'', '').replace('(', '').replace(')', ''))
