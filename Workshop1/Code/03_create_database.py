from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "Data" / "Raw" / "Sales_with_NaNs_v1.3.csv"
LIGHTGBM_DATA_PATH = BASE_DIR / "Data" / "Processed" / "Sales_with_NaNs_v1.3_imputed_lightgbm.csv"
DATABASE_PATH = BASE_DIR / "Data" / "Processed" / "workshop1_customer_data.db"
OUTPUT_DIR = BASE_DIR / "Output" / "Task1" / "03_relational_database"


def ReplaceNanWithNone(value):
    if pd.isna(value):
        return None
    return value


def SaveQuery(connection, query, output_path):
    result = pd.read_sql_query(query, connection)
    result.to_csv(output_path, index=False)
    return result


DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_data = pd.read_csv(RAW_DATA_PATH)
imputed_data = pd.read_csv(LIGHTGBM_DATA_PATH)

with sqlite3.connect(DATABASE_PATH) as connection:
    # Rådata og imputering holdes adskilt, så de oprindelige NULL stadig kan ses.
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS ImputedInterventionResults;
        DROP TABLE IF EXISTS ImputationMethods;
        DROP TABLE IF EXISTS InterventionResults;
        DROP TABLE IF EXISTS Customers;

        CREATE TABLE Customers (
            customer_id INTEGER PRIMARY KEY,
            group_name TEXT,
            customer_segment TEXT
        );

        CREATE TABLE InterventionResults (
            result_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            sales_before REAL,
            sales_after REAL,
            customer_satisfaction_before REAL,
            customer_satisfaction_after REAL,
            purchase_made TEXT,
            FOREIGN KEY (customer_id) REFERENCES Customers (customer_id)
        );

        CREATE TABLE ImputationMethods (
            method_id INTEGER PRIMARY KEY,
            method_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE ImputedInterventionResults (
            imputed_result_id INTEGER PRIMARY KEY,
            method_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            group_name TEXT,
            customer_segment TEXT,
            sales_before REAL,
            sales_after REAL,
            customer_satisfaction_before REAL,
            customer_satisfaction_after REAL,
            purchase_made TEXT,
            FOREIGN KEY (method_id) REFERENCES ImputationMethods (method_id),
            FOREIGN KEY (customer_id) REFERENCES Customers (customer_id)
        );
        """
    )
    connection.commit()

    customer_rows = []
    result_rows = []
    imputed_rows = []

    for row_index, row in raw_data.iterrows():
        customer_id = row_index + 1
        customer_rows.append(
            (
                customer_id,
                ReplaceNanWithNone(row["Group"]),
                ReplaceNanWithNone(row["Customer_Segment"]),
            )
        )
        result_rows.append(
            (
                customer_id,
                customer_id,
                ReplaceNanWithNone(row["Sales_Before"]),
                ReplaceNanWithNone(row["Sales_After"]),
                ReplaceNanWithNone(row["Customer_Satisfaction_Before"]),
                ReplaceNanWithNone(row["Customer_Satisfaction_After"]),
                ReplaceNanWithNone(row["Purchase_Made"]),
            )
        )

    for row_index, row in imputed_data.iterrows():
        customer_id = row_index + 1
        imputed_rows.append(
            (
                customer_id,
                1,
                customer_id,
                ReplaceNanWithNone(row["Group"]),
                ReplaceNanWithNone(row["Customer_Segment"]),
                ReplaceNanWithNone(row["Sales_Before"]),
                ReplaceNanWithNone(row["Sales_After"]),
                ReplaceNanWithNone(row["Customer_Satisfaction_Before"]),
                ReplaceNanWithNone(row["Customer_Satisfaction_After"]),
                ReplaceNanWithNone(row["Purchase_Made"]),
            )
        )

    connection.executemany(
        "INSERT INTO Customers (customer_id, group_name, customer_segment) VALUES (?, ?, ?);",
        customer_rows,
    )
    connection.executemany(
        """
        INSERT INTO InterventionResults (
            result_id,
            customer_id,
            sales_before,
            sales_after,
            customer_satisfaction_before,
            customer_satisfaction_after,
            purchase_made
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        result_rows,
    )
    connection.execute(
        "INSERT INTO ImputationMethods (method_id, method_name) VALUES (1, 'lightgbm');"
    )
    connection.executemany(
        """
        INSERT INTO ImputedInterventionResults (
            imputed_result_id,
            method_id,
            customer_id,
            group_name,
            customer_segment,
            sales_before,
            sales_after,
            customer_satisfaction_before,
            customer_satisfaction_after,
            purchase_made
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        imputed_rows,
    )
    connection.commit()

    # Små queries der viser, at tabellerne kan bruges relationelt.
    SaveQuery(
        connection,
        """
        SELECT
            c.group_name,
            COUNT(*) AS customers
        FROM Customers AS c
        GROUP BY c.group_name
        ORDER BY customers DESC;
        """,
        OUTPUT_DIR / "query_1_customers_by_group.csv",
    )
    SaveQuery(
        connection,
        """
        SELECT
            c.customer_segment,
            AVG(r.sales_before) AS avg_sales_before,
            AVG(r.sales_after) AS avg_sales_after,
            AVG(r.sales_after - r.sales_before) AS avg_sales_change
        FROM Customers AS c
        JOIN InterventionResults AS r
            ON c.customer_id = r.customer_id
        WHERE c.customer_segment IS NOT NULL
        GROUP BY c.customer_segment
        ORDER BY avg_sales_change DESC;
        """,
        OUTPUT_DIR / "query_2_sales_change_by_segment.csv",
    )
    SaveQuery(
        connection,
        """
        SELECT
            c.group_name,
            r.purchase_made,
            COUNT(*) AS rows
        FROM Customers AS c
        JOIN InterventionResults AS r
            ON c.customer_id = r.customer_id
        WHERE c.group_name IS NOT NULL
            AND r.purchase_made IS NOT NULL
        GROUP BY c.group_name, r.purchase_made
        ORDER BY c.group_name, r.purchase_made;
        """,
        OUTPUT_DIR / "query_3_purchase_counts_by_group.csv",
    )
    SaveQuery(
        connection,
        """
        SELECT
            c.customer_id,
            c.group_name,
            c.customer_segment,
            r.sales_before,
            r.sales_after,
            r.customer_satisfaction_before,
            r.customer_satisfaction_after,
            r.purchase_made
        FROM Customers AS c
        JOIN InterventionResults AS r
            ON c.customer_id = r.customer_id
        WHERE r.customer_satisfaction_after IS NULL
        LIMIT 20;
        """,
        OUTPUT_DIR / "query_4_rows_with_missing_satisfaction_after.csv",
    )
    SaveQuery(
        connection,
        """
        SELECT
            c.customer_id,
            r.customer_satisfaction_after AS raw_customer_satisfaction_after,
            i.customer_satisfaction_after AS lightgbm_imputed_customer_satisfaction_after
        FROM Customers AS c
        JOIN InterventionResults AS r
            ON c.customer_id = r.customer_id
        JOIN ImputedInterventionResults AS i
            ON c.customer_id = i.customer_id
        WHERE r.customer_satisfaction_after IS NULL
        LIMIT 20;
        """,
        OUTPUT_DIR / "query_5_raw_null_vs_lightgbm_imputed_values.csv",
    )

    counts = {
        "Customers": pd.read_sql_query("SELECT COUNT(*) AS rows FROM Customers;", connection).loc[0, "rows"],
        "InterventionResults": pd.read_sql_query("SELECT COUNT(*) AS rows FROM InterventionResults;", connection).loc[0, "rows"],
        "ImputationMethods": pd.read_sql_query("SELECT COUNT(*) AS rows FROM ImputationMethods;", connection).loc[0, "rows"],
        "ImputedInterventionResults": pd.read_sql_query("SELECT COUNT(*) AS rows FROM ImputedInterventionResults;", connection).loc[0, "rows"],
    }
    notes = [
        "Workshop 1 relational database",
        "",
        "Tables:",
        "- Customers(customer_id PRIMARY KEY, group_name, customer_segment)",
        "- InterventionResults(result_id PRIMARY KEY, customer_id FOREIGN KEY, sales_before, sales_after, customer_satisfaction_before, customer_satisfaction_after, purchase_made)",
        "- ImputationMethods(method_id PRIMARY KEY, method_name)",
        "- ImputedInterventionResults(imputed_result_id PRIMARY KEY, method_id FOREIGN KEY, customer_id FOREIGN KEY, imputed feature values)",
        "",
        "Rows:",
        *[f"- {table}: {int(count)}" for table, count in counts.items()],
        "",
        "Missing values from the CSV are stored as SQL NULL values.",
        "Only the LightGBM-imputed version is stored in the imputed results table.",
    ]
    (OUTPUT_DIR / "database_notes.txt").write_text("\n".join(notes), encoding="utf-8")
