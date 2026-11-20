import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// Served from https://science-of-model-organisms.github.io/mobench/ (GitHub Pages project site).
export default defineConfig({
  site: 'https://science-of-model-organisms.github.io',
  base: '/mobench',
  output: 'static',
  trailingSlash: 'always',
  vite: { plugins: [tailwindcss()] },
});
