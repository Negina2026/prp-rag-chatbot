from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from openai import OpenAI
import os
import re

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_FILE = "Inventory.csv"
KNOWLEDGE_FILE = "knowledge.txt"


def normalize_availability(value: str) -> str:
    value = str(value).strip().lower()

    mapping = {
        "current": "Available",
        "available": "Available",
        "remnant": "Limited availability",
        "limited availability": "Limited availability",
        "limited": "Limited availability",
        "stopped": "Not available",
        "not available": "Not available",
        "out of stock": "Not available",
    }

    return mapping.get(value, "Available")


def load_inventory():
    df = pd.read_csv(CSV_FILE)
    df = df.fillna("")
    df.columns = df.columns.astype(str).str.strip().str.lower()

    rename_map = {}
    for col in df.columns:
        if col == "base price":
            rename_map[col] = "price"
        elif col == "product status code":
            rename_map[col] = "availability"
        elif col == "status":
            rename_map[col] = "availability"

    df = df.rename(columns=rename_map)

    expected = [c for c in ["sku", "product", "price", "availability"] if c in df.columns]
    df = df[expected].copy()

    if "availability" in df.columns:
        df["availability"] = df["availability"].apply(normalize_availability)

    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price_num"] = None

    return df


def load_knowledge():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def is_inventory_question(user_query: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_query}".lower()

    inventory_keywords = [
        "wine", "wines", "price", "available", "availability", "in stock",
        "recommend", "sweet", "red", "white", "sparkling", "rose", "rosé",
        "moscato", "cabernet", "chardonnay", "merlot", "pinot", "malbec",
        "cheap", "cheapest", "expensive", "most expensive", "under", "below",
        "above", "cost", "bottle", "buy", "purchase", "premium", "option",
        "more", "yes please", "tell me more"
    ]
    return any(keyword in combined for keyword in inventory_keywords)


def infer_flags(df: pd.DataFrame):
    if "product" not in df.columns:
        df["is_sweet"] = False
        df["is_red"] = False
        df["is_white"] = False
        df["is_sparkling"] = False
        df["is_rose"] = False
        return df

    products = df["product"].astype(str).str.lower()

    sweet_terms = [
        "moscato", "spatlese", "auslese", "eiswein", "beerenauslese",
        "demi sec", "port", "framboise", "peche"
    ]
    red_terms = [
        "cabernet", "merlot", "pinot noir", "malbec", "barolo",
        "montepulciano", "pinotage", "tuscan", "dornfelder"
    ]
    white_terms = [
        "chardonnay", "sauvignon blanc", "chenin blanc", "bianco",
        "ribolla", "blanc", "fiano"
    ]
    sparkling_terms = ["sparkling", "brut", "cremant", "sekt", "secco"]
    rose_terms = ["rose", "rosé", "rosato"]

    df["is_sweet"] = products.apply(lambda x: any(term in x for term in sweet_terms))
    df["is_red"] = products.apply(lambda x: any(term in x for term in red_terms))
    df["is_white"] = products.apply(lambda x: any(term in x for term in white_terms))
    df["is_sparkling"] = products.apply(lambda x: any(term in x for term in sparkling_terms))
    df["is_rose"] = products.apply(lambda x: any(term in x for term in rose_terms))

    return df


def search_inventory(user_query: str, df: pd.DataFrame, max_results: int = 5):
    query = user_query.lower().strip()
    working_df = df.copy()

    if "availability" in working_df.columns:
        available_only = working_df[
            working_df["availability"].isin(["Available", "Limited availability"])
        ].copy()
        if not available_only.empty:
            working_df = available_only

    working_df = infer_flags(working_df)

    if any(term in query for term in ["most expensive", "highest price", "priciest"]):
        ranked = working_df.sort_values(by="price_num", ascending=False, na_position="last")
        return ranked.head(1).drop(
            columns=["price_num", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
            errors="ignore"
        ).to_dict(orient="records")

    if any(term in query for term in ["cheapest", "least expensive", "lowest price", "most affordable"]):
        ranked = working_df.sort_values(by="price_num", ascending=True, na_position="last")
        return ranked.head(1).drop(
            columns=["price_num", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
            errors="ignore"
        ).to_dict(orient="records")

    m_under = re.search(r"(under|below|less than)\s*\$?\s*(\d+(\.\d+)?)", query)
    if m_under:
        price_limit = float(m_under.group(2))
        working_df = working_df[working_df["price_num"] <= price_limit].copy()

    m_over = re.search(r"(over|above|more than)\s*\$?\s*(\d+(\.\d+)?)", query)
    if m_over:
        price_floor = float(m_over.group(2))
        working_df = working_df[working_df["price_num"] >= price_floor].copy()

    if "sweet" in query:
        working_df = working_df[working_df["is_sweet"]].copy()

    if re.search(r"\bred\b", query):
        working_df = working_df[working_df["is_red"]].copy()

    if re.search(r"\bwhite\b", query):
        working_df = working_df[working_df["is_white"]].copy()

    if "sparkling" in query or "brut" in query:
        working_df = working_df[working_df["is_sparkling"]].copy()

    if "rose" in query or "rosé" in query or "rosato" in query:
        working_df = working_df[working_df["is_rose"]].copy()

    if "premium" in query:
        working_df = working_df.sort_values(by="price_num", ascending=False, na_position="last")
        return working_df.head(max_results).drop(
            columns=["price_num", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
            errors="ignore"
        ).to_dict(orient="records")

    if "product" in working_df.columns:
        products = working_df["product"].astype(str).str.lower()
        direct = working_df[products.str.contains(re.escape(query), na=False)]
        if not direct.empty:
            return direct.head(max_results).drop(
                columns=["price_num", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
                errors="ignore"
            ).to_dict(orient="records")

    query_words = [
        w for w in re.findall(r"[a-zA-Z']+", query)
        if w not in {
            "do", "you", "have", "a", "an", "the", "any", "wine", "wines",
            "under", "below", "less", "than", "over", "above", "more",
            "available", "sweet", "red", "white", "sparkling", "brut",
            "rose", "rosé", "rosato", "recommend", "show", "me", "what",
            "is", "your", "most", "least", "highest", "lowest", "price",
            "cost", "buy", "purchase", "yes", "please"
        }
    ]

    if "product" in working_df.columns:
        def score_row(product_name: str) -> int:
            p = str(product_name).lower()
            return sum(1 for word in query_words if word in p)

        working_df["score"] = working_df["product"].apply(score_row)
        scored = working_df.sort_values(
            by=["score", "price_num"],
            ascending=[False, False],
            na_position="last"
        )

        positive = scored[scored["score"] > 0]
        if not positive.empty:
            return positive.head(max_results).drop(
                columns=["price_num", "score", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
                errors="ignore"
            ).to_dict(orient="records")

    return working_df.head(max_results).drop(
        columns=["price_num", "score", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
        errors="ignore"
    ).to_dict(orient="records")


def build_inventory_context(records):
    if not records:
        return "No matching inventory found."

    lines = []
    for item in records:
        lines.append(
            f"- {item.get('product', 'Unknown')} | "
            f"Price: ${item.get('price', 'N/A')} | "
            f"Availability: {item.get('availability', 'Not available')}"
        )

    return "\n".join(lines)


def search_knowledge(user_query: str, knowledge_text: str, max_chunks: int = 3):
    if not knowledge_text.strip():
        return ""

    paragraphs = [p.strip() for p in knowledge_text.split("\n\n") if p.strip()]
    query_words = [w.lower() for w in re.findall(r"[a-zA-Z']+", user_query) if len(w) > 2]

    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for word in query_words if word in para_lower)
        scored.append((score, para))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for _, chunk in scored[:max_chunks] if chunk.strip()]
    return "\n\n".join(top_chunks)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "PRP Wine chatbot API is running.",
        "health": "/health",
        "chat": "/chat"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get("message", "")).strip()
        history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "Message is required."}), 400

        history_text = " ".join(
            str(msg.get("content", ""))
            for msg in history
            if isinstance(msg, dict)
        )

        knowledge_text = load_knowledge()
        knowledge_context = search_knowledge(
            f"{history_text} {user_message}".strip(),
            knowledge_text
        )

        if is_inventory_question(user_message, history_text):
            df = load_inventory()

            effective_query = user_message
            if user_message.lower() in ["yes", "yes please", "tell me more", "more", "what else"]:
                effective_query = history_text + " premium recommendations"

            matches = search_inventory(effective_query, df)
            inventory_context = build_inventory_context(matches)

            system_prompt = f"""
You are a PRP Wine assistant.

Use the retrieved inventory as the primary source for product names, prices, and availability.
Use the PRP knowledge base as supporting background information when helpful.

Rules:
- Do not make up product names, prices, or availability.
- Never reveal internal stock counts or exact quantities.
- Use only these public-safe availability labels:
  - Available
  - Limited availability
  - Not available
- If the user gives a short follow-up like "yes please" or "tell me more",
  use the conversation history to continue naturally.
- If appropriate, recommend 2 to 4 matching wines.
- Keep answers concise and professional.
- If the knowledge base is weak, you may give a brief helpful general answer, but do not present it as a confirmed PRP-specific fact.

Retrieved inventory:
{inventory_context}

Retrieved PRP knowledge:
{knowledge_context}
"""
        else:
            matches = []
            system_prompt = f"""
You are a PRP Wine assistant.

Use the retrieved PRP knowledge below as your primary source.

Rules:
- If the answer is clearly supported by the PRP knowledge, answer confidently.
- If the exact answer is not directly stated in the PRP knowledge, give a brief helpful response based on general knowledge, but do NOT present it as a confirmed PRP-specific fact.
- When using general knowledge, make that clear with wording like:
  - "Generally..."
  - "In many cases..."
  - "Based on typical wine service practices..."
- Do not invent PRP policies, guarantees, pricing, or operational details that are not in the knowledge base.
- Keep answers concise, natural, and helpful.
- Avoid replying only with "I do not have that information available."
- If helpful, guide the user back to PRP’s known offerings such as exclusive wines, in-home tastings, virtual tastings, and personalized experiences.

Retrieved PRP knowledge:
{knowledge_context}
"""

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-6:]:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = str(msg.get("content", "")).strip()
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({
            "answer": answer,
            "matches": matches
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)