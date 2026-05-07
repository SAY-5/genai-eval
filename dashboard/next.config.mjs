/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  env: {
    GENAI_EVAL_API_URL: process.env.GENAI_EVAL_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
