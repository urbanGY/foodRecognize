# Food Recognize Model Project

## use dcnn

* tensorflow.compat.v1 사용
* conda activate capstone 에서 사용 가능

## Directory

    .                    
    ├── image                      # 이미지 관련 폴더
    │   ├── fileStructure          # 학습용 이미지 meta data json file
    │   ├── readFile.py            # 이미지 리사이즈 모듈
    │   └── resizeImg              # 학습용 이미지 보관 폴더
    │       └── images             # 이미지파일들 카테고리, 이름별로 분류
    │
    ├── model                      # 결과 모델 저장
    ├── imgFetch.py                # 이미지 경로 리스트, one hot encoding label 리스트 생성 모듈
    └── learn.py                   # 이미지 학습 모듈
        
