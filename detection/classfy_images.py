import os
import json
import shutil

def classify_images(image_dir, label_dir, target_dir):
    # 이미지 디렉토리에서 모든 파일 목록을 가져옵니다.
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # 각 이미지 파일에 대하여
    for image in images:
        # JSON 레이블 파일 경로 생성
        label_file = os.path.splitext(image)[0] + '.json'
        label_path = os.path.join(label_dir, label_file)

        # JSON 파일이 존재하는 경우
        if os.path.exists(label_path):
            # JSON 파일 열기
            with open(label_path, 'r') as file:
                data = json.load(file)
                # 레이블에 따라 대응하는 폴더로 이미지 이동
                label = data['FILE'][0]['ITEMS'][0]['SEGMENT'].lower()  # JSON에서 레이블 읽기, 소문자로 변환
                if label in ['대형차', '중형차', '소형차']:
                    # 대상 디렉토리에 레이블 폴더 생성
                    label_folder = os.path.join(target_dir, label)
                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

                    # 이미지 파일 이동
                    shutil.move(os.path.join(image_dir, image), os.path.join(label_folder, image))
                else:
                    print(f"Unknown label for {image}")
        else:
            print(f"Label file not found for {image}")

    print("Image classification completed.")
