/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      {
        source: '/upload',
        destination: '/trial/upload',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
