# Playing Card Detector & Rank/Suit Recognizer (OpenCV)

여러 장의 트럼프 카드가 포함된 이미지에서 **카드 영역을 검출**하고, 각 카드의 **Rank(숫자/문자)와 Suit(무늬)를 (조커제외) 인식**하는 프로젝트

## What it does
- 카드 후보 영역 검출(Contour 기반) → 4꼭짓점 사각형만 필터링
- Perspective Transform으로 카드 이미지를 **200×300**으로 정규화
- 좌상단 코너에서 **rank/suit 영역 분리** 후 이진화
- **AbsDiff(픽셀 차이 합)** 기반으로 rank/suit 템플릿과 매칭하여 최종 라벨 결정

## Pipeline (Core)
1. **Preprocess**: BGR→Gray → Gaussian Blur(7×7)
2. **Mask**: Otsu Threshold → Morphology Open(3×3)로 노이즈 제거
3. **Card Detection**: contour 추출 → area 필터(5000~1.5M) → approxPolyDP(eps=0.015×peri)로 4꼭짓점만 유지
4. **Warping**: 꼭짓점 정렬 → perspective transform → (200×300)으로 정규화
5. **Corner + Binary**: 좌상단 코너 crop → 4배 확대 → 동적 threshold(white_level - 30)
6. **Rank/Suit**: 큰 컨투어 선택 → rank(70×125), suit(70×100)로 resize
7. **Template Match**: score = sum(absdiff)/255 → score 최소 템플릿을 결과로 선택

## Key Parameters
- Blur: 7×7
- (디버그/보조) Canny: (100, 200), Dilate: 5×5
- Otsu Threshold + Morph Open(3×3)
- Contour area: 5000 ~ 1.5M
- approxPolyDP: 0.015 × perimeter
- Warp size: 200×300
- Corner zoom: ×4
- Corner threshold: `white_level - 30`

## Results 
- 다수 카드 입력에서 대부분 정상 인식
- 코너 추출 노이즈로 suit가 다른 무늬로 오인식되는 케이스(예: 클로버→다이아) 발생
- 템플릿 유사(10 vs Q)로 미스매치 가능

## Limitations
- 템플릿(AbsDiff) 방식은 **스케일/오염/약간의 변형**에도 민감
- Warp 과정에서 보간으로 글자가 흐려져 인식률 저하 가능
- 카드가 **겹치거나 부분 가려짐**이 크면 컨투어/코너 추출 실패 → 인식 불가
- 질감 있는 배경에서는 카드와 비슷한 패턴이 사각형으로 잡힐 수 있음(필터링으로 일부 완화)

## Tech
- Python, OpenCV (threshold/morphology/contour/perspective transform/template matching)

<img width="569" height="258" alt="image" src="https://github.com/user-attachments/assets/f0b23a91-a1df-4c09-a180-8d623882517b" />
<img width="600" height="165" alt="image" src="https://github.com/user-attachments/assets/251ec749-4a5b-4f67-ae15-ff73f85764f2" />

