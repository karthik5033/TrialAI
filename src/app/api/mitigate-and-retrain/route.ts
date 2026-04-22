import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { session_id } = body;

    if (!session_id) {
      return NextResponse.json(
        { error: "Missing required field: session_id" },
        { status: 400 }
      );
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/mitigate-and-retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id }),
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
