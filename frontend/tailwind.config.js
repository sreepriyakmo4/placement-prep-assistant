/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['DM Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Syne', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: '#0f0f13',
          1: '#16161d',
          2: '#1e1e28',
          3: '#262633',
        },
        accent: {
          DEFAULT: '#7c6af7',
          dim: '#5a4fc4',
          glow: 'rgba(124,106,247,0.15)',
        },
        emerald: {
          neon: '#00e5a0',
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s infinite',
        'thinking': 'thinking 1.4s infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(12px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(124,106,247,0.3)' },
          '50%': { boxShadow: '0 0 24px rgba(124,106,247,0.7)' },
        },
        thinking: {
          '0%, 80%, 100%': { opacity: 0.3, transform: 'scale(0.8)' },
          '40%': { opacity: 1, transform: 'scale(1)' },
        },
      },
    },
  },
  plugins: [],
}
