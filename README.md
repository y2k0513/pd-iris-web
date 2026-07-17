# 홍채 기준 PD 추정 웹앱

모바일 웹 카메라 또는 업로드 사진을 MediaPipe Face Landmarker로 분석해 양안 PD를 근사하는 데모입니다.

## 계산

- 홍채 중심 `468`, `473` 사이의 픽셀 거리: `PD_px`
- 좌우 홍채 경계점의 최대 직경 평균: `iris_px`
- 기준 홍채 지름 기본값: `11.7 mm`

```text
PD_mm = PD_px × 11.7 / iris_px
```

얼굴 자세가 정면에서 벗어난 사진, 좌우 홍채 검출 크기가 크게 다른 사진, 눈이 너무 작게 촬영된 사진은 재촬영 대상으로 분류합니다.

## 실행

Node.js 20 이상이 필요합니다.

```bash
npm install
npm run dev
```

브라우저에서 표시된 주소를 엽니다. PC의 `localhost`에서는 카메라를 사용할 수 있습니다.

### 완전 로컬 모델 사용

기본 상태에서는 공식 Google Storage의 Face Landmarker 모델을 최초 로드할 때 내려받습니다. 모델까지 프로젝트에 저장하려면:

```bash
npm run setup:model
npm run dev
```

그러면 `public/models/face_landmarker.task`가 사용되며 분석 과정과 모델 로딩 모두 로컬에서 동작합니다.

## 스마트폰 테스트

카메라 API는 보안 컨텍스트가 필요합니다.

- `localhost`: 허용
- 실제 배포: HTTPS 필요
- 같은 Wi-Fi의 단순 `http://192.168.x.x`: 브라우저에서 카메라가 차단될 수 있음

GitHub Pages, Cloudflare Pages, Vercel, Netlify 등에 정적 배포할 수 있습니다.

## 빌드

```bash
npm run build
npm run preview
```

빌드 결과는 `dist/`에 생성됩니다.

## 한계

- 모든 사용자의 홍채 지름을 11.7mm로 가정하므로 개인차에 따른 체계적 오차가 있습니다.
- 화면상의 얼굴 자세와 홍채 검출 오차가 결과에 영향을 줍니다.
- 안경 처방이나 의료 판단에 사용하는 실측 장비를 대체하지 않습니다.

## 기술

- MediaPipe Tasks Vision `0.10.35`
- Vite `7.3.6`
- 외부 유료 API 및 API 키 없음

## GitHub Pages 배포

1. 이 폴더를 GitHub 저장소에 올립니다.
2. 저장소 `Settings → Pages → Source`를 **GitHub Actions**로 설정합니다.
3. `main` 브랜치에 push하면 포함된 워크플로가 WASM과 모델을 준비하고 자동 배포합니다.

```bash
git init
git add .
git commit -m "Initial PD iris web app"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```
