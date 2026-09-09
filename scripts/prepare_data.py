"""
Data extraction and preprocessing script for AppleSupport conversations in twcs.csv.
Extracts root customer queries paired with Apple's first resolution response,
cleans Twitter artifacts, and prepares training data and retrieval index.
"""

import html
import json
import re
from pathlib import Path
import pandas as pd


def clean_tweet_text(text: str) -> str:
    """Clean raw Twitter text: unescape HTML, normalize whitespace, strip user handles."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.replace("\ufe0f", "").replace("\u200d", "")
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_reply_text(text: str) -> str:
    """Clean Apple's reply text for retrieval and reference grounding."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.replace("\ufe0f", "").replace("\u200d", "")
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_label_intent(text: str) -> str | None:
    """Heuristically assign intent for high-precision bootstrap dataset."""
    t = text.lower()

    if re.search(r"\b(cracked|shattered|broken screen|dropped.*phone|water damage|liquid|swollen|swelling|bulging|speaker.*buzz|mic.*not working|home button|power button|volume button|screen.*black.*won't turn on)\b", t):
        return "hardware_physical_damage"

    if re.search(r"\b(apple id|icloud|locked out|verification code|two factor|2fa|forgot password|charged|charge|refund|subscription|billed|billing|receipt|unauthorized|itunes bill|account disabled)\b", t):
        return "account_security_billing"

    if re.search(r"\b(battery|drain|draining|charge|charging|charger|percentage drops|dies at|overheating|overheat|hot to touch|cable)\b", t):
        return "battery_power_charging"

    if re.search(r"\b(wifi|wi-fi|bluetooth|airpods.*connect|no service|cellular|lte|data not working|disconnecting|airdrop|gps|hotspot|carrier)\b", t):
        return "connectivity_network"

    if re.search(r"\b(pre-order|preorder|shipping|delivery|tracking|order number|genius bar|appointment|in-store|store pickup|ups|fedex|reserved)\b", t):
        return "store_orders_reservations"

    if re.search(r"\b(update|updated|updating|ios 11|ios11|ios 12|ios10|software update|laggy|freezing|frozen|restart.*loop|rebooting|keyboard.*bug|capital i|stuck on apple logo)\b", t):
        return "software_update_os"

    if re.search(r"\b(app store|app crash|crashing|spotify|youtube|whatsapp|instagram|camera.*black|photos.*sync|apple music|itunes.*sync|notifications|imessage)\b", t):
        return "apps_media_features"

    return None


def extract_pairs(twcs_path: str, max_rows: int = 1500000) -> pd.DataFrame:
    print(f"Reading {twcs_path} up to {max_rows:,} rows...")
    chunk_size = 250000
    apple_replies = []
    inbound_map = {}

    total_rows = 0
    for chunk in pd.read_csv(twcs_path, chunksize=chunk_size):
        total_rows += len(chunk)

        inb = chunk[chunk["inbound"]]
        for _, row in inb.iterrows():
            inbound_map[row["tweet_id"]] = (
                row["text"],
                row["in_response_to_tweet_id"],
                row["author_id"],
                row["created_at"],
            )

        app = chunk[chunk["author_id"] == "AppleSupport"]
        apple_replies.append(
            app[["tweet_id", "in_response_to_tweet_id", "text", "created_at"]]
        )

        if total_rows >= max_rows:
            break

    df_replies = pd.concat(apple_replies, ignore_index=True)
    print(f"Loaded {len(df_replies)} AppleSupport replies. Matching with customer tweets...")

    records = []
    for _, row in df_replies.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if parent_id in inbound_map:
            cust_text, root_parent, author_id, cust_created = inbound_map[parent_id]
            is_root = pd.isna(root_parent)
            cleaned_cust = clean_tweet_text(cust_text)
            cleaned_reply = clean_reply_text(row["text"])

            if len(cleaned_cust) >= 15 and len(cleaned_reply) >= 15:
                intent = rule_label_intent(cleaned_cust)
                records.append(
                    {
                        "cust_tweet_id": int(parent_id),
                        "reply_tweet_id": int(row["tweet_id"]),
                        "customer_text": cleaned_cust,
                        "raw_customer_text": cust_text,
                        "brand_reply": cleaned_reply,
                        "raw_brand_reply": row["text"],
                        "is_root": is_root,
                        "inferred_intent": intent,
                        "created_at": cust_created,
                    }
                )

    df_out = pd.DataFrame(records)
    print(f"Extracted {len(df_out)} matched pairs ({df_out['is_root'].sum()} root conversation starters).")
    return df_out


def main():
    twcs_path = "data/twcs/twcs.csv"
    if not Path(twcs_path).exists():
        print(f"Error: {twcs_path} not found!")
        return

    df = extract_pairs(twcs_path, max_rows=1500000)

    df_root = df[df["is_root"]].copy()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    kb_records = []
    for _, r in df_root.iterrows():
        kb_records.append(
            {
                "id": str(r["cust_tweet_id"]),
                "customer_text": r["customer_text"],
                "reply": r["brand_reply"],
                "intent": r["inferred_intent"] or "general_support",
            }
        )

    kb_path = out_dir / "apple_knowledge_base.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_records[:6000], f, indent=2, ensure_ascii=False)
    print(f"Saved {min(len(kb_records), 6000)} knowledge base pairs to {kb_path}")

    labeled = df_root[df_root["inferred_intent"].notna()]
    print("\nIntent distribution in root queries:")
    print(labeled["inferred_intent"].value_counts())

    train_records = []
    for _, r in labeled.iterrows():
        train_records.append(
            {
                "text": r["customer_text"],
                "intent": r["inferred_intent"],
                "reply": r["brand_reply"],
            }
        )
    train_path = out_dir / "intent_training_data.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_records)} labeled training examples to {train_path}")


if __name__ == "__main__":
    main()
