import { NextRequest, NextResponse } from "next/server";

export async function GET(
  req: NextRequest,
  { params }: { params: { run_id: string } }
) {
  try {
    const run_id = params.run_id;
    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    
    const response = await fetch(`${fastApiUrl}/api/remediation/status/${run_id}`);
    
    const rawText = await response.text();
    let data;
    try {
      data = JSON.parse(rawText);
    } catch (e) {
      console.error("FastAPI did not return JSON:", rawText);
      return NextResponse.json(
        { error: "Backend returned invalid JSON", rawResponse: rawText.slice(0, 200) },
        { status: 500 }
      );
    }
    
    return NextResponse.json(data, { status: response.status });

  } catch (error: any) {
    console.error("Remediation status proxy error:", error);
    return NextResponse.json(
      { error: "Failed to forward request to backend" },
      { status: 500 }
    );
  }
}
