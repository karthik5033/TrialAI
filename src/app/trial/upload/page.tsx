"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Redirect: /trial/upload → /upload
 * The upload flow has been unified into a single page at /upload.
 */
export default function TrialUploadRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/upload");
  }, [router]);
  return null;
}
