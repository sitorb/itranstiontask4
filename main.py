from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from itertools import combinations
import re 
import pandas as pd
import yaml
import streamlit as st
import matplotlib.pyplot as plt
data_dir = Path(".")

BASE_DIR = Path(__file__).resolve().parent
STORE_NAMES = [f"STORE{i}" for i in range(1, 6)]
EUR_TO_USD = Decimal("1.20")

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_phone(value):
    if pd.isna(value):
        return ""

    return re.sub(r"\D", "", str(value))

def parse_money(value):
    if pd.isna(value):
        return None

    original = str(value).strip()
    if not original:
        return None

    is_euro = "€" in original or "eur" in original.lower()
    cleaned = re.sub(r"(?i)eur|usd", "", original)
    cleaned = cleaned.replace("€", "").replace("$", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = re.sub(r"[^0-9,.\-]", "", cleaned)

    if not cleaned or cleaned in {"-", ".", ","}:
        return None

    if "," in cleaned and "." not in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")

    try:
        amount = Decimal(cleaned)
        if not amount.is_finite():
            return None

        if is_euro:
            amount *= EUR_TO_USD

        return amount.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):
        return None



def money_to_float(value):
    if value is None:
        return None
    return float(value)


def load_store(store_name):
    store_dir = BASE_DIR / store_name

    users_path = store_dir / "users.csv"
    orders_path = store_dir / "orders.parquet"
    books_path = store_dir / "books.yaml"

    for path in (users_path, orders_path, books_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    users = pd.read_csv(users_path)
    orders = pd.read_parquet(orders_path)

    with open(books_path, "r", encoding="utf-8") as file:
        books_data = yaml.safe_load(file)

    books = pd.DataFrame(books_data)
    books.columns = [
        str(column).lstrip(":").strip()
        for column in books.columns
    ]

    return users, orders, books


def clean_data(users, orders, books):
    warnings = []

    users = users.copy()
    orders = orders.copy()
    books = books.copy()
    #i need to normalize the data
    users.columns = [str(c).strip().lower() for c in users.columns]
    orders.columns = [str(c).strip().lower() for c in orders.columns]
    books.columns = [str(c).strip().lower() for c in books.columns]

    required_user_columns = {
        "id", "name", "address", "phone", "email"
    }
    required_order_columns = {
        "user_id", "book_id", "quantity", "unit_price", "timestamp"
    }
    required_book_columns = {"id", "title", "author"}

    for label, frame, required in [
        ("users", users, required_user_columns),
        ("orders", orders, required_order_columns),
        ("books", books, required_book_columns),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} is missing required columns: {sorted(missing)}. "
                f"Actual columns: {list(frame.columns)}"
            )

    users["id"] = users["id"].astype("string").str.strip()
    orders["user_id"] = orders["user_id"].astype("string").str.strip()
    orders["book_id"] = orders["book_id"].astype("string").str.strip()
    books["id"] = books["id"].astype("string").str.strip()

    # i need to clean user fields
    for column in ["name", "address", "phone", "email"]:
        users[column] = users[column].astype("string").str.strip()

    before = len(users)
    users = users.drop_duplicates()
    warnings.append(f"Removed {before - len(users)} exact duplicate user rows.")

    before = len(orders)
    orders = orders.drop_duplicates()
    warnings.append(f"Removed {before - len(orders)} exact duplicate order rows.")

    before = len(books)
    books = books.drop_duplicates()
    warnings.append(f"Removed {before - len(books)} exact duplicate book rows.")

   
    orders["timestamp"] = pd.to_datetime(
        orders["timestamp"],
        errors="coerce",
        utc=True
    )

    invalid_dates = int(orders["timestamp"].isna().sum())
    if invalid_dates:
        warnings.append(
            f"Removed {invalid_dates} orders with invalid timestamps."
        )
        orders = orders.dropna(subset=["timestamp"])

 
    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="coerce"
    )

    valid_quantity = (
        orders["quantity"].notna()
        & (orders["quantity"] > 0)
        & (orders["quantity"] % 1 == 0)
    )

    invalid_quantity = int((~valid_quantity).sum())
    if invalid_quantity:
        warnings.append(
            f"Removed {invalid_quantity} orders with invalid quantities."
        )
        orders = orders.loc[valid_quantity].copy()

    orders["quantity"] = orders["quantity"].astype(int)

 
    orders["unit_price_usd"] = orders["unit_price"].apply(parse_money)

    invalid_prices = int(orders["unit_price_usd"].isna().sum())
    if invalid_prices:
        warnings.append(
            f"Removed {invalid_prices} orders with invalid prices."
        )
        orders = orders.dropna(subset=["unit_price_usd"])

    orders["paid_price"] = orders.apply(
        lambda row: money_to_float(
            (
                Decimal(str(row["unit_price_usd"]))
                * Decimal(int(row["quantity"]))
            ).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP
            )
        ),
        axis=1
    )

    orders["date"] = orders["timestamp"].dt.strftime("%Y-%m-%d")
    orders["year"] = orders["timestamp"].dt.year
    orders["month"] = orders["timestamp"].dt.month
    orders["day"] = orders["timestamp"].dt.day

    before = len(orders)
    orders = orders.dropna(subset=["user_id", "book_id"])
    orders = orders[
        orders["user_id"].ne("") & orders["book_id"].ne("")
    ].copy()

    if len(orders) != before:
        warnings.append(
            f"Removed {before - len(orders)} orders with missing IDs."
        )

    books = books.dropna(subset=["id", "author"])
    books = books[
        books["id"].ne("") & books["author"].astype(str).str.strip().ne("")
    ].copy()

    return users, orders, books, warnings

def normalize_email(value):
    return normalize_text(value)


def reconcile_users(users):

    users = users.copy().reset_index(drop=True)

    fields = ["name", "address", "phone", "email"]

    users["_name"] = users["name"].map(normalize_text)
    users["_address"] = users["address"].map(normalize_text)
    users["_phone"] = users["phone"].map(normalize_phone)
    users["_email"] = users["email"].map(normalize_email)

    parent = list(range(len(users)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    
    buckets = {}

    normalized_fields = {
        "name": "_name",
        "address": "_address",
        "phone": "_phone",
        "email": "_email",
    }

    for index, row in users.iterrows():
        for chosen_fields in combinations(fields, 3):
            values = tuple(
                row[normalized_fields[field]]
                for field in chosen_fields
            )
            if any(value == "" for value in values):
                continue

            key = (chosen_fields, values)

            if key in buckets:
                union(index, buckets[key])
            else:
                buckets[key] = index

    users["customer_group"] = [find(i) for i in range(len(users))]

    return users



def split_authors(value):
    if pd.isna(value):
        return tuple()

    authors = [
        re.sub(r"\s+", " ", name).strip()
        for name in str(value).split(",")
    ]
    authors = [name for name in authors if name]

    return tuple(sorted(set(authors), key=str.casefold))


def analyze_store(users, orders, books):
    reconciled_users = reconcile_users(users)

    books = books.copy()
    books["author_set"] = books["author"].apply(split_authors)
    books["author_set_label"] = books["author_set"].apply(
        lambda authors: ", ".join(authors) if authors else "Unknown"
    )

    daily_revenue = (
        orders.groupby("date", as_index=False)["paid_price"]
        .sum()
        .sort_values("date")
    )

    top_days = (
        daily_revenue.sort_values("paid_price", ascending=False)
        .head(5)
        .copy()
    )

    top_days["paid_price"] = top_days["paid_price"].round(2)

    sales = orders.merge(
        books[["id", "title", "author_set", "author_set_label"]],
        left_on="book_id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )

    sales["author_set_label"] = sales["author_set_label"].fillna(
        "Unknown author"
    )

    popular_authors = (
        sales.groupby("author_set_label", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "books_sold"})
        .sort_values(
            ["books_sold", "author_set_label"],
            ascending=[False, True]
        )
    )

    customer_lookup = reconciled_users[["id", "customer_group"]].copy()
    customer_lookup["id"] = customer_lookup["id"].astype("string")

    customer_orders = orders.merge(
        customer_lookup,
        left_on="user_id",
        right_on="id",
        how="left"
    )

    customer_orders["customer_group"] = customer_orders[
        "customer_group"
    ].fillna(customer_orders["user_id"].map(
        lambda value: f"unmatched:{value}"
    ))

    customer_spending = (
        customer_orders.groupby("customer_group", as_index=False)
        .agg(total_spending=("paid_price", "sum"))
        .sort_values("total_spending", ascending=False)
    )


    if not customer_spending.empty:
        best_group = customer_spending.iloc[0]["customer_group"]

        best_user_ids = sorted(
            customer_orders.loc[
                customer_orders["customer_group"] == best_group,
                "user_id"
            ].dropna().astype(str).unique().tolist()
        )

        best_spending = float(
            customer_spending.iloc[0]["total_spending"]
        )
    else:
        best_user_ids = []
        best_spending = 0.0

    unique_customers = int(reconciled_users["customer_group"].nunique())

    unique_author_sets = int(
        books.loc[books["author_set"].map(bool), "author_set"].nunique()
    )

    if not popular_authors.empty:
        max_sold = popular_authors["books_sold"].max()
        most_popular = popular_authors.loc[
            popular_authors["books_sold"] == max_sold,
            "author_set_label"
        ].tolist()
        most_popular_count = int(max_sold)
    else:
        most_popular = []
        most_popular_count = 0

    return {
        "users": reconciled_users,
        "orders": orders,
        "books": books,
        "daily_revenue": daily_revenue,
        "top_days": top_days,
        "popular_authors": popular_authors,
        "unique_customers": unique_customers,
        "unique_author_sets": unique_author_sets,
        "most_popular": most_popular,
        "most_popular_count": most_popular_count,
        "best_user_ids": best_user_ids,
        "best_spending": best_spending,
        "total_revenue": float(orders["paid_price"].sum()),
        "warnings": [],
    }

st.set_page_config(
    page_title="Store Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Store Intelligence Dashboard")
st.caption(
    "Revenue, customer identity reconciliation, and author performance"
)

selected_store = st.sidebar.selectbox(
    "Select a store",
    STORE_NAMES
)

st.sidebar.markdown("-")
st.sidebar.caption("Each store is analyzed independently.")

try:
    with st.spinner(f"Loading and analyzing {selected_store}..."):
        raw_users, raw_orders, raw_books = load_store(selected_store)

        users, orders, books, cleaning_warnings = clean_data(
            raw_users, raw_orders, raw_books
        )

        result = analyze_store(users, orders, books)
        result["warnings"] = cleaning_warnings

except Exception as error:
    st.error(f"Could not analyze {selected_store}: {error}")
    st.info(
        "Check that all three files exist and that the required "
        "Python dependencies are installed."
    )
    st.stop()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total revenue",
    f"${result['total_revenue']:,.2f}"
)

col2.metric(
    "Unique customers",
    f"{result['unique_customers']:,}"
)

col3.metric(
    "Unique author sets",
    f"{result['unique_author_sets']:,}"
)

col4.metric(
    "Books sold",
    f"{int(orders['quantity'].sum()):,}"
)

st.markdown("---")

overview_tab, authors_tab, customers_tab, quality_tab = st.tabs(
    ["Revenue overview", "Authors", "Customers", "Data quality"]
)


with overview_tab:
    left, right = st.columns([1, 1.5])

    with left:
        st.subheader("Top 5 days by revenue")

        if result["top_days"].empty:
            st.info("No valid daily revenue data is available.")
        else:
            display_top_days = result["top_days"].rename(
                columns={
                    "date": "Date",
                    "paid_price": "Revenue (USD)"
                }
            )
            st.dataframe(
                display_top_days,
                hide_index=True,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                result["top_days"]["date"],
                result["top_days"]["paid_price"]
            )
            ax.set_title("Top 5 Revenue Days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue (USD)")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with right:
        st.subheader("Daily revenue trend")

        daily = result["daily_revenue"].copy()

        if daily.empty:
            st.info("No daily revenue data is available.")
        else:
            daily["date"]=pd.to_datetime(daily["date"])
            daily=daily.sort_values("date")
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(
                daily["date"],
                daily["paid_price"],
                color="#2878B5",
                marker="o",
                linewidth=1.5
            )
            ax.set_title(f"{selected_store}: Daily Revenue")
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue (USD)")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate(rotation=35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("View daily revenue data"):
                display_daily = daily.rename(
                    columns={
                        "date": "Date",
                        "paid_price": "Revenue (USD)"
                    }
                )
                st.dataframe(
                    display_daily,
                    hide_index=True,
                    use_container_width=True
                )



with authors_tab:
    st.subheader("Most popular author set")

    if result["most_popular"]:
        st.success(
            f"{' | '.join(result['most_popular'])} — "
            f"{result['most_popular_count']:,} books sold"
        )
    else:
        st.info("No author sales data is available.")

    st.subheader("Author sets ranked by quantity sold")

    authors_table = result["popular_authors"].rename(
        columns={
            "author_set_label": "Author set",
            "books_sold": "Books sold"
        }
    )

    st.dataframe(
        authors_table,
        hide_index=True,
        use_container_width=True
    )

    if not result["popular_authors"].empty:
        chart_data = result["popular_authors"].head(10).sort_values(
            "books_sold", ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            chart_data["author_set_label"],
            chart_data["books_sold"]
        )
        ax.set_title("Top Author Sets by Books Sold")
        ax.set_xlabel("Books sold (quantity)")
        ax.set_ylabel("Author set")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


with customers_tab:
    st.subheader("Best customer by total spending")

    st.metric(
        "Total spending",
        f"${result['best_spending']:,.2f}"
    )

    st.write("Associated user IDs:")
    st.code(str(result["best_user_ids"]), language="python")

    st.caption(
        "Customer matching compares normalized name, address, phone "
        "and email fields. Review merged identities before submission."
    )

    st.subheader("Customer identity groups")

    grouped_users = (
        result["users"]
        .groupby("customer_group", as_index=False)
        .agg(
            user_ids=("id", lambda values: ", ".join(
                sorted(set(values.dropna().astype(str)))
            )),
            names=("name", lambda values: " | ".join(
                sorted(set(values.dropna().astype(str)))
            )),
            addresses=("address", lambda values: " | ".join(
                sorted(set(values.dropna().astype(str)))
            )),
            phones=("phone", lambda values: " | ".join(
                sorted(set(values.dropna().astype(str)))
            )),
            emails=("email", lambda values: " | ".join(
                sorted(set(values.dropna().astype(str)))
            )),
        )
    )

    st.dataframe(
        grouped_users.drop(columns=["customer_group"]),
        hide_index=True,
        use_container_width=True
    )

with quality_tab:
    st.subheader("Cleaning report")

    for warning in result["warnings"]:
        st.write(f"• {warning}")

    st.subheader("Dataset sizes")

    size_data = pd.DataFrame({
        "Dataset": ["Users", "Orders", "Books"],
        "Rows after cleaning": [
            len(result["users"]),
            len(result["orders"]),
            len(result["books"])
        ]
    })

    st.dataframe(
        size_data,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Orders preview")

    preview_columns = [
        "user_id", "book_id", "quantity",
        "unit_price", "unit_price_usd", "paid_price",
        "timestamp", "date", "year", "month", "day"
    ]

    st.dataframe(
        result["orders"][preview_columns].head(100),
        hide_index=True,
        use_container_width=True
    )

st.markdown("---")
st.caption(
    f"Store: {selected_store} | "
    "Currency: USD | "
    "Exchange rate: €1 = $1.20"
)
