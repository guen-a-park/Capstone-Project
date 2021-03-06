# Capstone-Project๐๏ธ

<img src="https://user-images.githubusercontent.com/77844152/170850973-a6d9197c-48f5-454f-83c1-0774e946af3b.png"  width="900" height="230">

 (2022.03.03~2022.05.30)  

[Report](https://drive.google.com/file/d/1O3djaE1sIkKqKMFZ2DDkFma04vOLrcwb/view?usp=sharing) / [Result Video](https://drive.google.com/file/d/1jtTDy1kF_WwoVZf0EnC5DHSWTQszGJoF/view?usp=sharing) / [Presentation](https://drive.google.com/file/d/1Rvjpv1Ab4Hc6qQEWCPy_vL7AzPkf9qzl/view?usp=sharing) 

ํ๋ก์ ํธ์ ๋ํ ์์ธํ ์ค๋ช์ ๋ณด๊ณ ์์ ๋น๋์ค๋ฅผ ํตํด ํ์ธํ  ์ ์๋ค. 


#  Abstract

๋ณธ ํ๋ก์ ํธ์์๋ Object Detection ์๊ณ ๋ฆฌ์ฆ YOLOv3์ ๋ผ์ฆ๋ฒ ๋ฆฌ ํ์ด๋ฅผ ํ์ฉํด ์คํ ๋ฐ์ด ํฌ๋ฉง ์ฐฉ์ฉ ๋จ์ ์นด๋ฉ๋ผ๋ฅผ ์ ์ํจ์ผ๋ก์จ ํจ๊ณผ์ ์ผ๋ก ํฌ๋ฉง ์ฐฉ์ฉ์ ๋๋ คํ  ์ ์๋ ์์คํ์ ์ ์ํ๋ค.

**Keywords** Raspberry Pi, YOLOv3, Text Detection, OCR  




# Environments

๋ณธ ํ๋ก์ ํธ๋ ์๋์ฝ๋ค๋ฅผ ์ด์ฉํ ๋ก์ปฌํ๊ฒฝ ๋๋ colab์์ ์คํ์ด ๊ฐ๋ฅํ๋ค.

- ๋ก์ปฌ์์ ์คํํ ๊ฒฝ์ฐ Python 3.7๋ก ๊ฐ์ํ๊ฒฝ์ ๋ง๋ค์ด์ฃผ๊ณ  ์๋ ๋ช๋ น์ด๋ฅผ ํตํด ํ์ํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ค์นํด์ค๋ค. 

```shell
conda create -n (your own env name) python=3.7
conda activate (your own env name)
git clone https://github.com/guen-a-park/Capstone-Project.git
cd Capstone-Project
pip install -r requirements.txt
```

๋ก์ปฌ ํ๊ฒฝ์์ tensorflow-gpu ๋ฅผ ์ด์ฉํ๊ธฐ ์ํด์๋  ๋ค์ ๋งํฌ๋ฅผ ์ฐธ๊ณ ํด ๋ณ๋์ ํ๊ฒฝ์ค์ ์ด ํ์ํ๋ค. (์๋์ฐ10 ๊ธฐ์ค) [Link](https://github.com/guen-a-park/Capstone-Project/blob/main/%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD%20%EC%84%B8%ED%8C%85%20%EB%AC%B8%EC%84%9C.pdf)

- colab์์ ์คํํ  ๊ฒฝ์ฐ ๋ณธ ํ๋ก์ ํธ๋ฅผ ๋๋ผ์ด๋ธ์์ clone ํด์ค ํ **for_colab** ํด๋์ *object_detection_yolo_keras.ipynb* ํ์ผ์ ํตํด ์ฝ๋์คํ์ฌ๋ถ๋ฅผ ํ์ธํ  ์ ์๋ค.



# File explanation

- **convert_xml_csv.py**

  > train image์ [labelImg](https://github.com/tzutalin/labelImg)๋ฅผ ํตํด ๋ง๋ค์ด์ค annotation ํ์ผ์ train์ ํ์ํ ํ๋์ csv๋ก ๋ง๋ค์ด์ฃผ๋ ์ฝ๋์ด๋ค. 
  >
  > ๊ทธ ๊ฒฐ๊ณผ์ธ helmet_anno.csv์์๋ train image ๊ฒฝ๋ก์ ํจ๊ป ๋ฐ์ด๋ฉ๋ฐ์ค์ label ์ ๋ณด๊ฐ ๋ค์ด๊ฐ๊ฒ๋๋ค.

- **augmentation.ipynb**

  > train image๊ฐ ๋ถ์กฑํ ๊ฒฝ์ฐ data augmentation์ ํตํด dataset์ ๋๋ ค์ค ์ ์๋ค. augmenataion์ ์ํ ๋ค์ํ ์ต์์ [imgaug](https://github.com/aleju/imgaug)์์ ํ์ธํ  ์ ์๋ค. 
  >
  
- **train.py**

  > custom data๋ฅผ ์ด์ฉํด ์๋ก์ด yolo model์ ๋ง๋ค๊ธฐ ์ํ ์ฝ๋์ด๋ค. log_dir์์ trainig ๋์ค ์์ฑ๋ ๋ชจ๋ธ ์ญ์ ํ์ธํ  ์ ์๋ค.
  >
  > **Note** : batch size์ ๋ฐ๋ผ(์ฌ์ด์ฆ๊ฐ ์ปค์ง์๋ก) ๋ฉ๋ชจ๋ฆฌ ์๋ฌ๊ฐ ๋ฐ์ํ  ์ ์๋ค. ๋ํ gpu ์ฑ๋ฅ์ ๋ฐ๋ผ ๋ก์ปฌ ์คํ์ด ์ด๋ ค์ธ ์ ์์ผ๋ฏ๋ก ํ์ํ๋ค๋ฉด ๊ตฌ๊ธ ๋๋ผ์ด๋ธ์์ annotation.csv๋ฅผ ๋ค์ ์์ฑํ๊ณ  colab์์ trainํด์ค๋ค.

- **yolo.py**

  > object detection์ ํ์ํ ์ฌ๋ฌ ๋ชจ๋์ ๋ด์ ํด๋์คํ์ผ์ด๋ค. ์ฌ์ง์์ ๋๋ ์์์์์ object detection ์ฌ๋ถ์ ๋ฐ๋ผ์ *detect_image* ํจ์ ๋๋ *detect_video* ํจ์๋ฅผ ํ์ธํ๋ค. ํจ์์์๋ ํ๋ก์ ํธ์ ํ์ํ ์ ๋ณด๋ฅผ returnํด์ค๋ค.

- **yolo_video.py** 

  > ์์ฑํ ๋ชจ๋ธ์ ํตํด ์ต์ข์ ์ผ๋ก ํ๋ก์ ํธ์ ์๊ณ ๋ฆฌ์ฆ์ ์คํํ  ์ ์๋ ์ฝ๋์ด๋ค. ์คํ ๊ณผ์ ์ ๋ค์๊ณผ ๊ฐ๋ค.
  >
  > - **train.py**๋ฅผ ํตํด์๋กญ๊ฒ ์์ฑํ ๋ชจ๋ธ๊ณผ ํ๋จํ  ํด๋์ค์ ๋ํ ์ ๋ณด๋ฅผ ๋ด์ ํ์ผ์ ๊ฒฝ๋ก๋ฅผ ์ง์ ํด์ค๋ค. ์ดํ testํ  ์ด๋ฏธ์ง์ ๊ฒฝ๋ก๋ ์ง์ ํ๋ค.
  >
  > - *detect_image*๋ฅผ ํตํด ์ฌ์ง์ ํด๋์ค๋ฅผ Helmet๊ณผ NonHelmet์ผ๋ก ํ๋จํ๊ณ  ๋ฐ์ด๋ฉ๋ฐ์ค ์ ๋ณด๋ฅผ ๋ฐ์์จ๋ค.
  > - ๋ง์ฝ ์คํ ๋ฐ์ด ํ์น์๊ฐ ํฌ๋ฉง์ ๋ฏธ์ฐฉ์ฉํ์ ๊ฒฝ์ฐ ํด๋น ๋ฐ์ด๋ฉ๋ฐ์ค๋ฅผ *textROI* ํจ์์ ์ ๋ฌํด์ค๋ค.
  > -  *textROI* ํจ์๋ text detection ๋ชจ๋ธ์ธ EAST text detector๋ฅผ ์ด์ฉํ์ฌ ์คํ ๋ฐ์ด ๋ฒํธํ์ ์์น๋ฅผ ํ์งํ๊ณ  ํด๋น ์ด๋ฏธ์ง๋ฅผ ๋ฐํํด์ค๋ค.
  > - *textRead* ํจ์์์๋ ๋ฒํธํ ์ด๋ฏธ์ง์์ ๋ฌธ์๋ฅผ ์ถ์ถํ์ฌ ๋์งํธ๋ฌธ์๋ก ๋ฐํํด์ค๋ค. [์ต์](https://muthu.co/all-tesseract-ocr-options/)์ ํตํด ์ฌ์ฉํ๋ ๋ชจ๋ธ ๋ฐ ์ธ์ ์ธ์ด๋ฅผ ๋ฐ๊ฟ์ค ์ ์๋ค.  



# Contributors



<table>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/63901494/129583717-42d19759-7586-4de0-aea9-5e935295f4dd.png" width="100" height="100"><br /><sub><b>๊น๊ท๋ฆฌ</b></sub></td>
    <td align="center"><a href="https://github.com/mina-kim-1015"><img src="https://avatars.githubusercontent.com/u/79397445?v=4" width="100" height="100"><br /><sub><b>๊น๋ฏผ์</b></sub></td>
     <td align="center"><img src="https://avatars.githubusercontent.com/u/79395493?v=4" width="100" height="100"><br /><sub><b>๊นํ๋ฏผ</b></sub></td>
    <td align="center"><a href="https://github.com/guen-a-park"><img src="https://avatars.githubusercontent.com/u/77844152?s=400&v=4" width="100" height="100"><br /><sub><b>๋ฐ๊ทผ์</b></sub></td>
    <td align="center"><a href="https://github.com/hong-ep"><img src="https://avatars.githubusercontent.com/u/104953860?v=4" width="100" height="100"><br /><sub><b>ํ์ํ</b></sub></td>
  </tr>
</table>





# Reference  

- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- [tzutalin/labelImg](https://github.com/tzutalin/labelImg)
- [aleju/imgaug](https://github.com/aleju/imgaug)
- [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- [ํ์๋ ํธ ์ค์น](https://ddolcat.tistory.com/954)
- [EAST text detector](https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)



