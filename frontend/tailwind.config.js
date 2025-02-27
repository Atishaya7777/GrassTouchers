/** @type {import('tailwindcss').Config} */

// import { animations, boxShadow, colors, fonts, keyframes } from './src/theme';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      keyframes: {
        // ...keyframes,
      },
      animation: {
        // ...animations,
      },
      fontFamily: {
        // ...fonts,
      },
      colors: {
        // ...colors,
      },
      boxShadow: {
        // ...boxShadow,
      },
    },
  },
  plugins: [],
};
