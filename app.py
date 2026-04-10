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


def load_inventory():
    df = pd.read_csv(CSV_FILE)
    df = df.fillna("")
    df.columns = df.columns.astype(str).str.strip()

    if "Product" not in df.columns:
        raw_df = pd.read_csv(CSV_FILE, header=None)
        raw_df = raw_df.fillna("")

        header_row_index = None
        for i, row in raw_df.iterrows():
            row_values = [str(x).strip() for x in row.tolist()]
            if "Product" in row_values and "Available" in row_values:
                header_row_index = i
                break

        if header_row_index is not None:
            df = pd.read_csv(CSV_FILE, header=header_row_index)
            df = df.fillna("")
            df.columns = df.columns.astype(str).str.strip()

    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c == "sku":
            rename_map[col] = "sku"
        elif c == "product":
            rename_map[col] = "product"
        elif c in ["base price", "price"]:
            rename_map[col] = "price"
        elif c == "available":
            rename_map[col] = "available"
        elif c in ["product status code", "status"]:
            rename_map[col] = "status"

    df = df.rename(columns=rename_map)

    expected = ["sku", "product", "price", "available", "status"]
    existing = [c for c in expected if c in df.columns]
    df = df[existing].copy()

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

    if "available" in df.columns:
        df["available_num"] = pd.to_numeric(df["available"], errors="coerce")
    else:
        df["available_num"] = None

    print("COLUMNS:", df.columns.tolist())
    return df


def infer_flags(df: pd.DataFrame):
    if "product" not in df.columns:
        df["is_sweet"] = False
        df["is_red"] = False
        df["is_white"] = False
        df["is_sparkling"] = False
        df["is_rose"] = False
        return df

    sweet_terms = [
        "moscato", "spatlese", "auslese", "eiswein", "beerenauslese",
        "demi sec", "mild", "port", "framboise", "peche", "raspberry",
        "blue", "cream", "sekt"
    ]
    red_terms = [
        "cabernet", "merlot", "pinot noir", "malbec", "montepulciano",
        "primitivo", "barolo", "sangiovese", "cannonau", "tuscan",
        "vino nobile", "pinotage", "jumilla", "taurasi", "bordeaux blend"
    ]
    white_terms = [
        "chardonnay", "sauvignon blanc", "chenin blanc", "bianco",
        "gruner veltliner", "ribolla gialla", "torbato", "fiano",
        "kabinett", "orange wine", "mosel"
    ]
    sparkling_terms = [
        "sparkling", "brut", "cremant", "sekt", "secco"
    ]
    rose_terms = [
        "rose", "rosato"
    ]

    products = df["product"].astype(str).str.lower()

    df["is_sweet"] = products.apply(lambda x: any(term in x for term in sweet_terms))
    df["is_red"] = products.apply(lambda x: any(term in x for term in red_terms))
    df["is_white"] = products.apply(lambda x: any(term in x for term in white_terms))
    df["is_sparkling"] = products.apply(lambda x: any(term in x for term in sparkling_terms))
    df["is_rose"] = products.apply(lambda x: any(term in x for term in rose_terms))

    return df


def search_inventory(user_query: str, df: pd.DataFrame, max_results: int = 5):
    query = user_query.lower().strip()
    working_df = df.copy()

    if "status" in working_df.columns:
        working_df = working_df[working_df["status"].astype(str).str.lower() == "current"].copy()

    working_df = infer_flags(working_df)

    price_limit = None
    price_floor = None

    m_under = re.search(r"(under|below|less than)\s*\$?\s*(\d+(\.\d+)?)", query)
    if m_under:
        price_limit = float(m_under.group(2))

    m_over = re.search(r"(over|above|more than)\s*\$?\s*(\d+(\.\d+)?)", query)
    if m_over:
        price_floor = float(m_over.group(2))

    if price_limit is not None and "price_num" in working_df.columns:
        working_df = working_df[working_df["price_num"] <= price_limit].copy()

    if price_floor is not None and "price_num" in working_df.columns:
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

    query_words = [
        w for w in re.findall(r"[a-zA-Z']+", query)
        if w not in {
            "do", "you", "have", "a", "an", "the", "any", "wine", "wines",
            "under", "below", "less", "than", "over", "above", "more",
            "current", "available", "sweet", "red", "white", "sparkling",
            "brut", "rose", "rosato", "recommend", "show", "me"
        }
    ]

    if "product" in working_df.columns:
        def score_row(product_name: str) -> int:
            p = str(product_name).lower()
            return sum(1 for word in query_words if word in p)

        working_df["score"] = working_df["product"].apply(score_row)

        sort_columns = []
        ascending = []

        if "score" in working_df.columns:
            sort_columns.append("score")
            ascending.append(False)

        if "available_num" in working_df.columns:
            sort_columns.append("available_num")
            ascending.append(False)

        scored = working_df.sort_values(by=sort_columns, ascending=ascending) if sort_columns else working_df

        positive = scored[scored["score"] > 0] if "score" in scored.columns else pd.DataFrame()

        if not positive.empty:
            return positive.head(max_results).drop(
                columns=["price_num", "available_num", "score", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
                errors="ignore"
            ).to_dict(orient="records")

        if not scored.empty and (
            "sweet" in query or
            re.search(r"\bred\b", query) or
            re.search(r"\bwhite\b", query) or
            "sparkling" in query or
            "brut" in query or
            "rose" in query or
            "rosé" in query or
            "rosato" in query or
            price_limit is not None or
            price_floor is not None
        ):
            return scored.head(max_results).drop(
                columns=["price_num", "available_num", "score", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
                errors="ignore"
            ).to_dict(orient="records")

        fallback = scored.head(max_results)
        return fallback.drop(
            columns=["price_num", "available_num", "score", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
            errors="ignore"
        ).to_dict(orient="records")

    return working_df.head(max_results).drop(
        columns=["price_num", "available_num", "is_sweet", "is_red", "is_white", "is_sparkling", "is_rose"],
        errors="ignore"
    ).to_dict(orient="records")


def build_inventory_context(records):
    if not records:
        return "No matching inventory found."

    lines = []
    for item in records:
        line = (
            f"- {item.get('product', 'Unknown')} | "
            f"Price: ${item.get('price', 'N/A')} | "
            f"Available Units: {item.get('available', 'N/A')} | "
            f"Status: {item.get('status', 'N/A')}"
        )
        lines.append(line)

    return "\n".join(lines)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    df = load_inventory()
    matches = search_inventory(user_message, df)
    inventory_context = build_inventory_context(matches)

    system_prompt = f"""
You are a PRP Wine assistant.

Answer using the retrieved inventory below.
Do not make up product names, prices, stock, or wine characteristics beyond what can reasonably be inferred from the product names.
If the requested product type is not found in the retrieved inventory, say so clearly.
If appropriate, recommend 2 to 4 matching products.
Prefer products with status "Current".
Mention price and availability in a concise way.

Retrieved inventory:
{inventory_context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.4
    )

    answer = response.choices[0].message.content

    return jsonify({
        "answer": answer,
        "matches": matches
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)