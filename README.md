# 홍채·동공 기준 PD 추정 웹앱

모바일 웹 내부 전면 카메라, MediaPipe Face Landmarker, OpenCV.js를 이용해 양안 PD를 근사하는 데모입니다.

## 최종 계산

1. MediaPipe의 얼굴·눈·홍채 랜드마크로 좌우 눈 ROI를 찾습니다.
2. 고해상도 눈 crop을 최대 4배 확대합니다.
3. OpenCV.js에서 grayscale → gamma/히스토그램 보정 → blur → 어두운 영역 threshold → morphology → contour → ellipse fitting 순서로 동공 후보를 찾습니다.
4. 동공 신뢰도가 0.8 이상이면 OpenCV 중심을 사용하고, 0.5~0.8이면 OpenCV 70% + MediaPipe 30%로 융합합니다. 0.5 미만이면 MediaPipe 홍채 중심으로 fallback합니다.
5. 홍채 지름은 MediaPipe 홍채 경계점으로 계산하며, 최종 PD는 다음 2D 비율식을 사용합니다.

```text
PD_mm = 동공 중심 간 픽셀 거리 × 기준 홍채 지름 / 평균 홍채 픽셀 지름
```

기준 홍채 지름 기본값은 `11.7mm`입니다. MediaPipe 상대 z 기반 3D 추정값은 최종 PD에 섞지 않고 원근 이상과 촬영 품질 검증에만 사용합니다.

## 촬영 흐름

1. 웹에서 가능한 최고 해상도의 전면 카메라를 요청합니다.
2. 브라우저가 지원하면 `ImageCapture.takePhoto()`로 고해상도 정지 사진을 얻고, 지원하지 않으면 비디오 프레임으로 fallback합니다.
3. 얼굴 자세, 프레이밍, 시선, 원근 대칭, 홍채 크기, 눈 영역 선명도와 밝기를 실시간 검사합니다.
4. `촬영 준비`를 누르고 모든 조건을 1초 동안 유지하면 100ms 간격으로 3장을 자동 촬영합니다.
5. 각 프레임에서 동공 중심을 정밀 보정하고, 품질을 통과한 프레임들의 PD 중앙값을 최종 raw PD로 사용합니다.
6. 성별 분포 soft prior를 적용한 보정 PD와 원본 raw PD를 모두 표시합니다.

## 기본 분포 prior

```text
남성: 64~70mm, 중심 67mm, scale 3mm
여성: 58~64mm, 중심 61mm, scale 3mm
```

## 실행

Node.js 20 이상이 필요합니다.

```bash
npm install
npm run setup:assets
npm run dev
```

`public/opencv/opencv.js`가 포함되어 있으면 `setup:opencv`는 다운로드를 건너뜁니다.

## 확인 및 빌드

```bash
npm test
npm run build
```

## 한계

- 모든 사용자의 홍채 지름을 같은 값으로 가정하므로 개인차에 따른 체계적 오차가 있습니다.
- OpenCV 기반 동공 검출은 반사광, 컬러렌즈, 짙은 속눈썹, 안경, 낮은 조도에서 실패할 수 있으며 이때 MediaPipe 중심으로 fallback합니다.
- MediaPipe z는 절대 mm 깊이가 아닌 단안 상대 깊이입니다.
- 안경 처방이나 의료 판단에 사용하는 실측 장비를 대체하지 않습니다.

## GitHub Pages 배포

1. 변경 파일을 GitHub 저장소에 push합니다.
2. 저장소 `Settings → Pages → Source`를 **GitHub Actions**로 설정합니다.
3. `main` 브랜치에 push하면 모델·OpenCV 자산 확인, 테스트, 빌드 후 자동 배포됩니다.
