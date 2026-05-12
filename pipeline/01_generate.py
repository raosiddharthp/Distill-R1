import json, yaml, argparse, re
from pathlib import Path
from openai import OpenAI

def load_config(path):
    return yaml.safe_load(open(path))

def chunk_text(text, size=1500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def generate_sample(client, model, chunk):
    prompt = f"""You are a domain expert assistant. Given the following product documentation,
generate a realistic customer query and a detailed, accurate answer that includes relevant
policy citations, SKU details, and actionable next steps.

Document:
{chunk}

Respond ONLY with a valid JSON object with keys: "instruction", "output", "reasoning_steps".
No markdown, no backticks, just raw JSON."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/synthetic/train.jsonl")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = OpenAI(
        api_key=cfg["teacher"]["api_key"],
        base_url="https://openrouter.ai/api/v1"
    )

    text = Path(args.input).read_text()
    chunks = chunk_text(text)[:args.num_samples]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0

    with open(args.output, "w") as f:
        for i, chunk in enumerate(chunks):
            try:
                raw = generate_sample(client, cfg["teacher"]["model"], chunk)
                obj = json.loads(raw)
                f.write(json.dumps(obj) + "\n")
                written += 1
                print(f"[{i+1}/{len(chunks)}] OK")
            except Exception as e:
                skipped += 1
                print(f"[{i+1}/{len(chunks)}] SKIP — {e}")

    print(f"\nDone. {written} written, {skipped} skipped → {args.output}")

if __name__ == "__main__":
    main()
