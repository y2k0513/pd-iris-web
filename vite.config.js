import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages처럼 하위 경로에 배포해도 public 자산을 찾도록 상대 경로 사용
  base: './',
  server: {
    host: true,
  },
});
