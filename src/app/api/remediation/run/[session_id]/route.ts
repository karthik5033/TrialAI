import { NextRequest, NextResponse } from "next/server";

export async function POST(
  req: NextRequest,
  { params }: { params: { session_id: string } }
) {
  try {
    const session_id = params.session_id;
    const body = await req.json();

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/api/remediation/run/${session_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

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
    console.error("Remediation run proxy error:", error);
    return NextResponse.json(
      { error: "Failed to forward request to backend" },
      { status: 500 }
    );
  }
}
