import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0B1017",
          900: "#0F1620",
          800: "#161F2C",
          700: "#202B3B",
          600: "#334157",
        },
        mist: "#C9D2DE",
        paper: "#E8ECF2",
        teal: {
          400: "#4FD1C5",
          500: "#2FBBAC",
        },
        amber: "#E0A458",
      },
      fontFamily: {
        display: ["Georgia", "serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
