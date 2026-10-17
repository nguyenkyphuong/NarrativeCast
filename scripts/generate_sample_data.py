"""
scripts/generate_sample_data.py
Generates a realistic synthetic sample dataset matching the MADOC schema,
specifically focused on Reddit Technology (r/technology) as defined in docs/problem_specification.md.
Saves to data/sample/madoc_sample.parquet.
"""

import os
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def generate_sample_data(num_samples: int = 4800, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    start_epoch = 1704067200  # 2024-01-01 00:00:00 UTC
    total_days = 60           # 60 days to allow multi-horizon evaluation (h = 3, 7, 14 days)
    end_epoch = start_epoch + total_days * 86400

    user_pool = [f"usr_{uuid.uuid4().hex[:8]}" for _ in range(400)]

    # Narratives within Reddit r/technology (distinguishing narrative from topic as per problem_specification.md)
    narrative_templates = {
        "AI_Regulation_Stifles_Innovation": {
            "topic": "AI Governance",
            "keywords": ["ai regulation", "stifle innovation", "startup barrier", "licensing red tape", "tech competition"],
            "texts": [
                "Overly strict AI compliance laws will entrench big tech monopolies and crush young startups.",
                "Mandatory government model licensing acts as an innovation barrier for European and US tech firms.",
                "Proposed bureaucratic guardrails on open-weight machine learning threaten independent open research.",
                "Congress hearings demonstrate regulators fundamentally misunderstand open-source developer workflows.",
                "Startups warn that heavy algorithmic liability penalties will drive technical talent offshore.",
                "Excessive compliance overhead slows down vital enterprise product shipping."
            ],
            # Moderate initially, gains major traction around day 25-45 as policy hearings intensify
            "prob_func": lambda day: 0.08 + 0.35 * (1 / (1 + np.exp(-(day - 30) / 4.0)))
        },
        "AI_Safety_Demands_Urgent_Guardrails": {
            "topic": "AI Governance",
            "keywords": ["ai safety", "urgent guardrails", "existential risk", "deepfake disinformation", "model audit"],
            "texts": [
                "Frontier artificial intelligence systems require mandatory independent third-party red-teaming audits.",
                "Uncontrolled deployment of deepfakes and automated social bots threatens democratic election integrity.",
                "Leading computer scientists call for binding safety covenants before training next-generation frontier models.",
                "Federal oversight is necessary to prevent automated cybersecurity exploits and chemical bio-risk misuse.",
                "Public safety must take precedence over fast corporate commercialization of autonomous agents.",
                "Ethics benchmarks show current alignment techniques are inadequate for high-stakes decision agents."
            ],
            # Surges early in weeks 2-4, then stabilizes at steady high discussion
            "prob_func": lambda day: 0.10 + 0.30 * np.exp(-((day - 20) ** 2) / 120.0)
        },
        "Big_Tech_Antitrust_Breakup": {
            "topic": "Market Competition",
            "keywords": ["antitrust", "break up tech", "monopoly abuse", "app store tax", "ftc litigation"],
            "texts": [
                "Department of Justice antitrust lawsuit details anticompetitive exclusive distribution agreements.",
                "Ecosystem lock-in and high app store commissions prevent healthy platform price competition.",
                "Regulators argue structural divestiture is the only remedy to restore open digital search markets.",
                "Tech monopolists systematically acquire nascent emerging competitors to neutralize competitive threats.",
                "Consumer welfare is harmed by predatory default placement and gatekeeper platform power.",
                "Court rulings indicate renewed regulatory appetite to enforce traditional antitrust laws in tech."
            ],
            # Steady presence across entire timeline
            "prob_func": lambda day: 0.16 + 0.04 * np.sin(day / 5.0)
        },
        "Open_Source_AI_Democratization": {
            "topic": "Open Source Tech",
            "keywords": ["open source ai", "democratize models", "local llm", "hugging face", "community weights"],
            "texts": [
                "Local inference on consumer hardware proves open community models can rival proprietary APIs.",
                "Open-weights foster global transparency, academic scrutiny, and decentralized technological agency.",
                "Developer community releases highly optimized quantized weights running efficiently on laptops.",
                "Fine-tuning specialized smaller models outperforms generalized closed-source foundation models.",
                "Decentralized training collectives challenge centralized corporate computing dominance.",
                "Open source libraries empower developers to build without vendor lock-in or recurring API costs."
            ],
            # Strong steady gradual upward adoption trend
            "prob_func": lambda day: 0.12 + 0.003 * day
        },
        "Critical_Zero_Day_Infrastructure_Breach": {
            "topic": "Cybersecurity",
            "keywords": ["zero day exploit", "cloud infrastructure breach", "cve vulnerability", "supply chain attack", "kernel exploit"],
            "texts": [
                "Critical remote code execution vulnerability discovered in widespread open-source logging daemon.",
                "Emergency patch advisory issued as threat actors actively exploit unauthenticated gateway flaws.",
                "Massive cloud infrastructure intrusion exposes cryptographic tokens across enterprise instances.",
                "Security teams execute around-the-clock incident response to rotate compromised API credentials.",
                "CISA adds severe kernel flaw to known exploited vulnerabilities catalog with 48-hour deadline.",
                "Forensic audit reveals state-sponsored threat group maintained persistent access for months."
            ],
            # Acute dramatic spike around days 38-42
            "prob_func": lambda day: 0.03 + 0.65 * np.exp(-((day - 40) ** 2) / 4.0)
        },
        "Semiconductor_Foundry_Sovereignty": {
            "topic": "Hardware & Supply Chain",
            "keywords": ["semiconductor chips", "foundry fabrication", "chips act subsidies", "euv lithography", "wafer capacity"],
            "texts": [
                "Multi-billion dollar semiconductor fabrication facility breaks ground under federal chips subsidies.",
                "Geopolitical export curbs on advanced high-NA EUV lithography equipment reshape global chip packaging.",
                "Leading chip designer secures priority production allocation for next-generation 2nm wafer nodes.",
                "Foundry expansion delays highlight structural shortage of specialized cleanroom engineering talent.",
                "Automakers and electronics manufacturers diversify supplier contracts to mitigate geopolitical supply risks.",
                "Domestic fabrication packaging initiatives aim to insulate strategic hardware from supply shocks."
            ],
            # Fading: very high in first 10 days, then decreases
            "prob_func": lambda day: 0.35 * np.exp(-day / 12.0) + 0.04
        }
    }

    records = []
    parent_posts = []

    days = np.random.uniform(0, total_days, size=num_samples)
    days.sort()

    for idx, day in enumerate(days):
        ts = int(start_epoch + day * 86400 + random.randint(0, 3600))
        post_id = f"madoc_red_{uuid.uuid4().hex[:12]}"
        user_id = random.choice(user_pool)
        platform = "reddit"
        community = "technology"  # Primary focus as per problem_specification.md Section 2

        # Sample narrative based on temporal dynamics at this day
        narrative_keys = list(narrative_templates.keys())
        narrative_weights = [narrative_templates[k]["prob_func"](day) for k in narrative_keys]
        total_w = sum(narrative_weights)
        norm_weights = [w / total_w for w in narrative_weights]
        chosen_narrative = random.choices(narrative_keys, weights=norm_weights)[0]

        # Interaction type: ~45% POST, ~50% COMMENT, ~5% REPOST
        if len(parent_posts) > 10 and random.random() < 0.55:
            interaction_type = "COMMENT"
            parent = random.choice(parent_posts[-40:])
            parent_id = parent["post_id"]
            parent_user_id = parent["user_id"]
        else:
            interaction_type = "POST"
            parent_id = None
            parent_user_id = None

        template_text = random.choice(narrative_templates[chosen_narrative]["texts"])
        keyword_token = random.choice(narrative_templates[chosen_narrative]["keywords"])
        topic_name = narrative_templates[chosen_narrative]["topic"]

        if interaction_type == "COMMENT":
            content = f"Regarding {keyword_token}: {template_text} This is crucial for r/technology."
        else:
            content = f"[{topic_name}] {template_text} Discussion around {keyword_token}."

        # Sentiment
        if "Breach" in chosen_narrative:
            sentiment = float(np.clip(np.random.normal(-0.60, 0.20), -1.0, 1.0))
        elif "Democratization" in chosen_narrative:
            sentiment = float(np.clip(np.random.normal(0.45, 0.25), -1.0, 1.0))
        elif "Stifles_Innovation" in chosen_narrative:
            sentiment = float(np.clip(np.random.normal(-0.25, 0.35), -1.0, 1.0))
        elif "Safety" in chosen_narrative:
            sentiment = float(np.clip(np.random.normal(-0.15, 0.30), -1.0, 1.0))
        elif "Antitrust" in chosen_narrative:
            sentiment = float(np.clip(np.random.normal(-0.20, 0.30), -1.0, 1.0))
        else:
            sentiment = float(np.clip(np.random.normal(0.10, 0.30), -1.0, 1.0))

        url = f"https://reddit.com/r/technology/comments/{uuid.uuid4().hex[:6]}" if random.random() < 0.4 else ""
        strict_filter = 1 if ("Breach" in chosen_narrative or "Safety" in chosen_narrative) and random.random() < 0.25 else 0

        record = {
            "post_id": post_id,
            "publish_date": ts,
            "user_id": user_id,
            "parent_id": parent_id,
            "parent_user_id": parent_user_id,
            "content": content,
            "url": url,
            "language": "en",
            "interaction_type": interaction_type,
            "platform": platform,
            "community": community,
            "sentiment_vader": round(sentiment, 4),
            "strict_filter": strict_filter,
            "topic": topic_name,
            "ground_truth_narrative": chosen_narrative
        }
        records.append(record)
        if interaction_type == "POST":
            parent_posts.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "madoc_sample.parquet")
    print("Generating sample dataset aligned with problem_specification.md (Reddit r/technology, 60 days)...")
    df = generate_sample_data(num_samples=5000)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {datetime.fromtimestamp(df['publish_date'].min(), tz=timezone.utc)} to {datetime.fromtimestamp(df['publish_date'].max(), tz=timezone.utc)}")
    print("Narrative counts:")
    print(df["ground_truth_narrative"].value_counts())
