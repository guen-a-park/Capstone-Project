# annotation과 image 디렉토리 설정. annotation디렉토리에 있는 파일 확인. 
import os
from pathlib import Path

HOME_DIR = str(Path.home())
print(HOME_DIR)

ANNO_DIR = 'C:/Users/kate1/capstone/train_image/annotation'
IMAGE_DIR = 'C:/Users/kate1/capstone/train_image/image'

files = os.listdir(ANNO_DIR)
print('파일 개수는:',len(files))  # 200개
print(files)  # Dicrectory에 있는 파일 이름 전부가 list 변수에 저장된다.

#multiclass labeling 위함
det = {}
det['Helmet'] = 0
det['NonHelmet']=1

import glob
import xml.etree.ElementTree as ET

def xml_to_csv(path, output_filename):
    """
    path : annotation Detectory
    filename : ouptut file name
    """
    xml_list = []
    # xml 확장자를 가진 모든 파일의 절대 경로로 xml_file할당. 
    with open(output_filename, "w") as train_csv_file:
        for xml_file in glob.glob(path + '/*.xml'):
            # path에 있는 xml파일중 하나 하나를 가져온다. 
            tree = ET.parse(xml_file) 
            root = tree.getroot()
            # 파일내에 있는 모든 object Element를 찾음. 
            full_image_name = os.path.join(IMAGE_DIR, root.find('filename').text)
            value_str_list = ' '
            # find all <object>인것 다 찾는다
            for obj in root.findall('object'): 
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                # helmet, nonhelmet
                label = obj.find('name').text

                if label in det:
                    class_id = det[label]
                else:
                    class_id = -1

                #class_id = 0
                value_str = ('{0},{1},{2},{3},{4}').format(x1, y1, x2, y2, class_id)
                value_str_list = value_str_list+value_str+' ' 
                # box1 box2 ......
                # object별 정보를 tuple형태로 object_list에 저장. 
            train_csv_file.write(full_image_name+' '+ value_str_list+'\n') # image_file_path box1 box2 ... boxN \n ... image_file_path
        # xml file 찾는 for loop 종료 

xml_to_csv(ANNO_DIR, os.path.join(ANNO_DIR,'helmet_anno.csv'))
print(os.path.join(ANNO_DIR,'helmet_anno.csv'))