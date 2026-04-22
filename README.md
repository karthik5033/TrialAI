# TrialAI — AI Fairness Courtroom
> Put your AI on trial before the world does

## What is TrialAI?

The widespread deployment of AI models across critical sectors—hiring, lending, healthcare, and criminal justice—has introduced an unprecedented level of scale and efficiency. However, it has simultaneously codified and amplified historical biases at a frightening velocity. Too often, models are shipped as black boxes with structural prejudices hidden beneath high accuracy scores, leading to devastating real-world consequences and massive legal liabilities for enterprises.

TrialAI is an adversarial, multi-agent AI courtroom designed to drag algorithmic bias into the light. We transform abstract statistical metrics into an interactive, high-stakes audit where specialized AI agents—acting as the Prosecution, Defense, and Judge—aggressively debate the evidence of bias within your dataset. By pitting LLMs against your model's predictions, TrialAI identifies proxy variables, disparate impact violations, and unequal outcomes *before* you deploy.

Our platform bridges the gap between data science and ethical oversight. With deep integration of SHAP values, counterfactual simulators, and live synthetic juries, TrialAI makes fairness explainable, actionable, and mathematically verifiable. It isn't just a dashboard; it's a clinical stress test for model integrity.

## Live Demo
- **Demo**: `http://localhost:3000/demo`
- **Start Trial**: `http://localhost:3000/trial/upload`

## How It Works
1. 📁 **Upload any CSV dataset**: Provide your training data and define the target prediction column and sensitive attributes (e.g., race, gender, age).
2. ⚖️ **Three AI agents debate bias evidence in real time**: The Prosecution aggressively highlights statistical disparities, the Defense argues for model validity and lack of intent, and an impartial Judge delivers a comprehensive verdict.
3. 👥 **Synthetic jury of 12 personas experiences the model live**: Watch as a diverse, synthetic jury is subjected to your model's predictions, providing immediate, human-centric visibility into adverse impacts.
4. 📄 **Receive verdict, reform order and downloadable audit report**: Obtain a binding Court Reform Order detailing concrete mitigation steps and export an official PDF audit report.

## Key Features
- Multi-agent adversarial courtroom (Prosecution/Defense/Judge)
- Real fairness metrics (Demographic Parity, Equal Opportunity, Disparate Impact)
- COMPAS criminal justice dataset built in
- Counterfactual bias simulation
- Bias Risk Score (0-100)
- Downloadable PDF audit report
- Works on any CSV dataset

## Tech Stack
- Next.js 14, TypeScript, Tailwind CSS, Framer Motion
- FastAPI Python microservice
- Groq API (Llama 3)
- Recharts, shadcn/ui, Lucide
- scikit-learn, pandas, numpy

## Getting Started
**Prerequisites**: Node.js 18+, Python 3.9+, Groq API key, Gemini API key

**Step 1 — Clone and install**:
```bash
git clone https://github.com/your-username/trialai.git
cd trialai
npm install
```

**Step 2 — Environment variables**:
```bash
cp .env.example .env.local
```
Add your `GROQ_API_KEY` and `GEMINI_API_KEY` to the `.env.local` file.

**Step 3 — Start FastAPI**:
```bash
cd fastapi
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Step 4 — Start Next.js**:
```bash
# In a new terminal window at the project root
npm run dev
```

Open `http://localhost:3000`

## Use Cases
- **HR**: Audit hiring models for gender/race bias
- **Finance**: Test loan approval models for demographic fairness
- **Healthcare**: Check diagnostic models for age/ethnicity bias
- **Criminal Justice**: Audit recidivism prediction tools

## The COMPAS Case

In 2016, an independent investigation revealed that the COMPAS algorithm—a risk assessment tool used nationwide by judges to determine bail and sentencing—was structurally biased against African American defendants, falsely flagging them as high-risk at nearly twice the rate of white defendants. This watershed moment exposed the grave dangers of deploying opaque algorithmic systems in high-stakes environments without rigorous ethical auditing. TrialAI utilizes the COMPAS dataset as its flagship demo to demonstrate how our adversarial audit could have caught these disparities before they destroyed lives.

## Hackathon
Built at Prompthathon — Domain 4: Unbiased AI Decision Making

## License
MIT
