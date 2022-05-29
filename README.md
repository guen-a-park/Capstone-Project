# Capstone-Project🏍️

<img src="https://user-images.githubusercontent.com/77844152/170850973-a6d9197c-48f5-454f-83c1-0774e946af3b.png"  width="900" height="230">

 (2022.03.03~2022.05.30)  

[Report]() / [Result Video](https://drive.google.com/file/d/1jtTDy1kF_WwoVZf0EnC5DHSWTQszGJoF/view?usp=sharing) / [Presentation](https://drive.google.com/file/d/1Rvjpv1Ab4Hc6qQEWCPy_vL7AzPkf9qzl/view?usp=sharing) 

프로젝트에 대한 자세한 설명은 보고서와 비디오를 통해 확인할 수 있다. 


#  Abstract

본 프로젝트에서는 Object Detection 알고리즘 YOLOv3와 라즈베리 파이를 활용해 오토바이 헬멧 착용 단속 카메라를 제작함으로써 효과적으로 헬멧 착용을 독려할 수 있는 시스템을 제안한다.

**Keywords** Raspberry Pi, YOLOv3, Text Detection, OCR  




# Environments

본 프로젝트는 아나콘다를 이용한 로컬환경 또는 colab에서 실행이 가능하다.

- 로컬에서 실행할경우 Python 3.7로 가상환경을 만들어주고 아래 명령어를 통해 필요한 라이브러리를 설치해준다. 

```shell
conda create -n (your own env name) python=3.7
conda activate (your own env name)
git clone https://github.com/guen-a-park/Capstone-Project.git
cd Capstone-Project
pip install -r requirements.txt
```

로컬 환경에서 tensorflow-gpu 를 이용하기 위해서는  다음 링크를 참고해 별도의 환경설정이 필요하다. (윈도우10 기준) [Link](https://github.com/guen-a-park/Capstone-Project/blob/main/%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD%20%EC%84%B8%ED%8C%85%20%EB%AC%B8%EC%84%9C.pdf)

- colab에서 실행할 경우 본 프로젝트를 드라이브에서 clone 해준 후 **for_colab** 폴더의 *object_detection_yolo_keras.ipynb* 파일을 통해 코드실행여부를 확인할 수 있다.



# File explanation

- **convert_xml_csv.py**

  > train image와 [labelImg](https://github.com/tzutalin/labelImg)를 통해 만들어준 annotation 파일을 train에 필요한 하나의 csv로 만들어주는 코드이다. 
  >
  > 그 결과인 helmet_anno.csv에서는 train image 경로와 함께 바운딩박스와 label 정보가 들어가게된다.

- **augmentation.ipynb**

  > train image가 부족한 경우 data augmentation을 통해 dataset을 늘려줄 수 있다. augmenataion을 위한 다양한 옵션은 [imgaug](https://github.com/aleju/imgaug)에서 확인할 수 있다. 
  >
  
- **train.py**

  > custom data를 이용해 새로운 yolo model을 만들기 위한 코드이다. log_dir에서 trainig 도중 생성된 모델 역시 확인할 수 있다.
  >
  > **Note** : batch size에 따라(사이즈가 커질수록) 메모리 에러가 발생할 수 있다. 또한 gpu 성능에 따라 로컬 실행이 어려울 수 있으므로 필요하다면 구글 드라이브에서 annotation.csv를 다시 생성하고 colab에서 train해준다.

- **yolo.py**

  > object detection에 필요한 여러 모듈을 담은 클래스파일이다. 사진에서 또는 영상에서의 object detection 여부에 따라서 *detect_image* 함수 또는 *detect_video* 함수를 확인한다. 함수에서는 프로젝트에 필요한 정보를 return해준다.

- **yolo_video.py** 

  > 생성한 모델을 통해 최종적으로 프로젝트의 알고리즘을 실행할 수 있는 코드이다. 실행 과정은 다음과 같다.
  >
  > - **train.py**를 통해새롭게 생성한 모델과 판단할 클래스에 대한 정보를 담은 파일의 경로를 지정해준다. 이후 test할 이미지의 경로도 지정한다.
  >
  > - *detect_image*를 통해 사진의 클래스를 Helmet과 NonHelmet으로 판단하고 바운딩박스 정보를 받아온다.
  > - 만약 오토바이 탑승자가 헬멧을 미착용했을 경우 해당 바운딩박스를 *textROI* 함수에 전달해준다.
  > -  *textROI* 함수는 text detection 모델인 EAST text detector를 이용하여 오토바이 번호판의 위치를 탐지하고 해당 이미지를 반환해준다.
  > - *textRead* 함수에서는 번호판 이미지에서 문자를 추출하여 디지털문자로 반환해준다. [옵션](https://muthu.co/all-tesseract-ocr-options/)을 통해 사용하는 모델 및 인식 언어를 바꿔줄 수 있다.  



# Contributors



<table>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/63901494/129583717-42d19759-7586-4de0-aea9-5e935295f4dd.png" width="100" height="100"><br /><sub><b>김규리</b></sub></td>
    <td align="center"><a href="https://github.com/mina-kim-1015"><img src="https://avatars.githubusercontent.com/u/79397445?v=4" width="100" height="100"><br /><sub><b>김민아</b></sub></td>
     <td align="center"><img src="https://avatars.githubusercontent.com/u/79395493?v=4" width="100" height="100"><br /><sub><b>김혜민</b></sub></td>
    <td align="center"><a href="https://github.com/guen-a-park"><img src="https://avatars.githubusercontent.com/u/77844152?s=400&v=4" width="100" height="100"><br /><sub><b>박근아</b></sub></td>
    <td align="center"><a href="https://github.com/hong-ep"><img src="https://avatars.githubusercontent.com/u/104953860?v=4" width="100" height="100"><br /><sub><b>홍은표</b></sub></td>
  </tr>
</table>





# Reference  

- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- [tzutalin/labelImg](https://github.com/tzutalin/labelImg)
- [aleju/imgaug](https://github.com/aleju/imgaug)
- [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- [테서렉트 설치](https://ddolcat.tistory.com/954)
- [EAST text detector](https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)



