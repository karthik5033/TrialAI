import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { session_id, target_column, sensitive_attributes, strategy } = body;

    if (!session_id || !target_column || !sensitive_attributes) {
      return NextResponse.json(
        { error: "Missing required fields: session_id, target_column, or sensitive_attributes" },
        { status: 400 }
      );
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/api/remediation/run/${session_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column, sensitive_attributes, strategy: strategy || "reweighing" }),
    });

    const rawText = await response.text();
    let data;
    try {
      data = JSON.parse(rawText);
    } catch (e) {
      console.error("FastAPI did not return JSON:", rawText);
      return NextResponse.json({ error: "Backend returned invalid JSON", rawResponse: rawText.slice(0, 200) }, { status: 500 });
    }
    return NextResponse.json(data, { status: response.status });

  } catch (error: any) {
    console.error("Mitigate-and-retrain proxy error:", error);
    return NextResponse.json(
      { error: "Failed to forward request to retrain service" },
      { status: 500 }
    );
  }
}
