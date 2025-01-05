import numpy as np
import json



labels_train = np.load('/home/zhouyifan/test/HLEG-main/data/intentonomy/train_label_vectors_intentonomy2020.npy').astype(np.float64)
with open('/home/zhouyifan/test/HLEG-main/data/intentonomy/intentonomy_train2020.json', 'r') as f:
    annos_dict_train = json.load(f)

labels_test = np.load('/home/zhouyifan/test/HLEG-main/data/intentonomy/val_label_vectors_intentonomy2020.npy').astype(np.float64)
with open('/home/zhouyifan/test/HLEG-main/data/intentonomy/intentonomy_val2020.json', 'r') as f:
    annos_dict_test = json.load(f)

lost_imgae = {'309bf6e625518372de94e4737c433d4a',
              '567c0b902f7e1a1c2df85102cb8be1f3',
              '309bf6e625518372de94e4737c433d4a',
              'ae174cd6fe17d7b990dbacb5e2393ea0',
              '2e0e95c5f339dc276bc618ea1ba99d30',
              '2548291faf3103f36083391aa29dad5b',
              'ba212b7db8f744b6583b0e839697e0dd',
              '7046374cec16eaf8887c25283617447f',
              'a3801db137f250f6ab929cd76ca1b2b7',
              '5872fd258fc6bb453a8f1886a47d1049',

              '61ccfd54a4a09bf768ecaefff438dc1b',
              '5f7d4be83144216ab5f2f6256c37a69d',
              'a46793304208ae05b15626898a6dff82',
              '7b8d12d0a7434f889a463c831aa09598',
              'cf123641970bfe968e05502ff9aa7855',
              'f51e34a98db003416bc4f0a2d79f0ae3',
              '5c282af0aadb15b9473350c4972c7444',
              '0b058ddcec06a37f506791304e7f5c05',
              'bba1911c16fab97cd512d22c427221a7',
              '711f3f4be3f4de2ad4e17925b23b460d',
              '0a06d5eabf8b255999e552f48baea4c9',
              '4f85eeee16e74bb055e91b92d8398a7d',
              '33990a4fde89ae006ead7fd3359fccb1',
              '6f878c261c57fde91438d97ae44b160d',
              'b134565e457f2f8475e6fcfb3f6e6a1c',
              '0201b01bd3604f6289ac7d7892065e14',
              '89a7b6ebaf49f3e42d74503ad1a33154'

              }


print(len(labels_train))
print(len(annos_dict_train['annotations']))

for i in range(len(labels_train) - 1, -1, -1):
    annos_i = annos_dict_train['annotations'][i]
    if annos_i['image_id'] in lost_imgae:

        print(annos_i['image_id'])
        labels_train = np.delete(labels_train, i, 0)
        del annos_dict_train['annotations'][i]

print(len(labels_train))
print(len(annos_dict_train['annotations']))


np.save('/home/zhouyifan/test/HLEG-main/data/intentonomy/train_label_vectors_intentonomy2020.npy', labels_train)

with open('/home/zhouyifan/test/HLEG-main/data/intentonomy/intentonomy_train2020.json', 'w') as f:
    json.dump(annos_dict_train,f)

print(len(labels_test))
print(len(annos_dict_test['annotations']))

for i in range(len(labels_test) - 1, -1, -1):
    annos_i = annos_dict_test['annotations'][i]
    if annos_i['image_id'] in lost_imgae:

        print(annos_i['image_id'])
        labels_test = np.delete(labels_test, i, 0)
        del annos_dict_test['annotations'][i]

print(len(labels_test))
print(len(annos_dict_test['annotations']))


np.save('/home/zhouyifan/test/HLEG-main/data/intentonomy/val_label_vectors_intentonomy2020.npy', labels_test)

with open('/home/zhouyifan/test/HLEG-main/data/intentonomy/intentonomy_val2020.json', 'w') as f:
    json.dump(annos_dict_test,f)