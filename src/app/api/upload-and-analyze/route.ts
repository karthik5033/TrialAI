import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    
    // Ensure the required fields exist, though we mainly just forward it
    const file = formData.get("file");
    const targetColumn = formData.get("target_column");
    const sensitiveAttributes = formData.get("sensitive_attributes");

    if (!file || !targetColumn || !sensitiveAttributes) {
      return NextResponse.json(
        { error: "Missing required fields: file, target_column, sensitive_attributes" },
        { status: 400 }
      );
    }

    // Forward the form data directly to FastAPI
    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/upload-and-analyze`, {
      method: "POST",
      body: formData,
    });

    // Check if response is ok, but we want to forward the response regardless
    const data = await response.json();
    
    return NextResponse.json(data, { status: response.status });

  } catch (error: any) {
    console.error("Error in upload-and-analyze route:", error);
    return NextResponse.json(
      { error: "Failed to forward request to analysis service" },
      { status: 500 }
    );
  }
}
