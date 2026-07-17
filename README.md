# 홍채 기준 PD 추정 웹앱

모바일 웹 내부 전면 카메라와 MediaPipe Face Landmarker를 사용해 양안 PD를 근사하는 데모입니다.

## 최종 계산

- 홍채 중심 `468`, `473` 사이의 2D 픽셀 거리: `PD_px`
- 좌우 홍채 경계점의 최대 직경 평균: `iris_px`
- 기준 홍채 지름 기본값: `11.7 mm`

```text
PD_mm = PD_px × 11.7 / iris_px
```

최종값은 반복성이 더 좋은 2D 홍채 비율만 사용합니다. MediaPipe 상대 z 기반 3D 추정값은 최종 PD에 섞지 않고 원근 이상과 촬영 품질 검증에 사용합니다.

## 촬영 흐름

1. 웹에서 고해상도 전면 카메라를 시작합니다.
2. `촬영 준비` 버튼을 누릅니다.
3. 얼굴 자세, 프레이밍, 시선, 원근 대칭 조건을 맞춥니다.
4. 조건을 1초 동안 유지하면 자동 촬영합니다.
5. 촬영된 고해상도 프레임을 다시 분석해 최종 결과를 표시합니다.

기본 자세 허용치는 Yaw 6°, Pitch 7°, Roll 5°이며 설정 화면에서 조절할 수 있습니다.

## 실행

Node.js 20 이상이 필요합니다.

```bash
npm install
npm run setup:model
npm run dev
```

## 확인 및 빌드

```bash
npm test
npm run build
```

## 한계

- 모든 사용자의 홍채 지름을 11.7mm로 가정하므로 개인차에 따른 체계적 오차가 있습니다.
- MediaPipe z는 절대 mm 깊이가 아닌 단안 상대 깊이입니다.
- 안경 처방이나 의료 판단에 사용하는 실측 장비를 대체하지 않습니다.

## GitHub Pages 배포

1. 변경 파일을 GitHub 저장소에 push합니다.
2. 저장소 `Settings → Pages → Source`를 **GitHub Actions**로 설정합니다.
3. `main` 브랜치에 push하면 자동 배포됩니다.
