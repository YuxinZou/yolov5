import json
import glob
import os
from tqdm import tqdm

path = 'D:/data/前海/标注数据/labelme部分/NH-03-0079_001_2022-05-26-12-00-00_2022-05-26-12-10-00'

add_people = [
    {
      "label": "1",
      "points": [
        [
          1153.1746031746031,
          432.53968253968253
        ],
        [
          1335.7142857142858,
          550.7936507936508
        ]
      ],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "1",
      "points": [
        [
          1412.121212121212,
          404.2424242424242
        ],
        [
          1545.4545454545453,
          497.57575757575756
        ]
      ],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
    }
]
for root, dirs, files in os.walk(path):
    for name in tqdm(files):
        if name.endswith('jpg'):
            fname = os.path.join(root, name)
            json_file = fname.replace('jpg', 'json')

            fr = open(json_file, 'r')
            data = json.load(fr)
            fr.close()
            print(data)
            fw = open(json_file, 'w')
            data['shapes'] += add_people
            data["imageData"] = None
            json.dump(data, fw, indent=2)
            fw.close()
