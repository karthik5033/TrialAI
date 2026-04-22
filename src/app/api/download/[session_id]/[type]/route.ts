import { NextRequest, NextResponse } from "next/server";

export async function GET(
  req: NextRequest,
  { params }: { params: { session_id: string; type: string } }
) {
  try {
    const { session_id, type } = params;

    if (!["model", "script"].includes(type)) {
      return NextResponse.json({ error: "Invalid download type" }, { status: 400 });
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const response = await fetch(`${fastApiUrl}/download/${session_id}/${type}`);

    if (!response.ok) {
      const data = await response.json();
      return NextResponse.json(data, { status: response.status });
    }

    const blob = await response.blob();
    const filename = type === "model" ? "mitigated_model.pkl" : "modified_training_script.py";

    return new NextResponse(blob, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });

  } catch (error: any) {
    console.error("Download proxy error:", error);
    return NextResponse.json(
      { error: "Failed to download file" },
      { status: 500 }
    );
  }
}
