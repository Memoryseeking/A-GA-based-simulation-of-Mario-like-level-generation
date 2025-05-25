import json
import os

json_files = ['transforms_train.json', 'transforms_val.json', 'transforms_test.json']
base_dir = '/home/zhaojun/Luminance-GS/Luminance-GS/data/jskm2'

for name in json_files:
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        print(f"❌ {name} 不存在，跳过")
        continue

    with open(path, 'r') as f:
        data = json.load(f)

    for frame in data['frames']:
        filename = os.path.basename(frame['file_path'])  # 只取文件名
        frame['file_path'] = filename

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ 修复完成: {name}")
