import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const file = formData.get("file");
    const modelFile = formData.get("model_file");
    const targetColumn = formData.get("target_column");
    const sensitiveAttributes = formData.get("sensitive_attributes");

    if (!file || !modelFile || !targetColumn || !sensitiveAttributes) {
      return NextResponse.json(
        { error: "Missing required fields: file, model_file, target_column, sensitive_attributes" },
        { status: 400 }
      );
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/api/analysis/full-analysis`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });

  } catch (error: any) {
    console.error("Full analysis proxy error:", error);
    return NextResponse.json(
      { error: "Failed to forward request to analysis service" },
      { status: 500 }
    );
  }
}
