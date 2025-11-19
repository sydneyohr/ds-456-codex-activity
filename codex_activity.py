import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_PATH = Path("data/Snow_Emergency_Howard_Tags_2024.csv")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Howard Snow Emergency tags dataset.
    Uses utf-8-sig to handle the BOM on the first column.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Clean up column names (strip spaces, keep original spelling)
    df.columns = [c.strip() for c in df.columns]

    # Parse date-time column if present
    if "Day_of_Dat" in df.columns:
        df["Day_of_Dat"] = pd.to_datetime(df["Day_of_Dat"], errors="coerce")

    return df


def basic_summaries(df: pd.DataFrame) -> None:
    """Print basic information and simple summaries."""
    print("=== Basic structure ===")
    print("Shape (rows, columns):", df.shape)
    print("\nColumn types:")
    print(df.dtypes)

    print("\nFirst five rows:")
    print(df.head())

    print("\n=== Numeric summary ===")
    print(df.describe(include="number"))

    # Frequency tables for some key categorical columns if they exist
    for col in ["Community", "Neighborho", "Street", "Day", "Tow_Zone", "Snow_Eme_2"]:
        if col in df.columns:
            print(f"\nTop values for {col}:")
            print(df[col].value_counts().head(10))


def plot_tags_by_neighborhood(df: pd.DataFrame) -> None:
    """Bar plot of tag counts by neighborhood (top 15)."""
    if "Neighborho" not in df.columns:
        return

    counts = df["Neighborho"].value_counts().head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, orient="h", color="steelblue")
    plt.xlabel("Number of tags")
    plt.ylabel("Neighborhood")
    plt.title("Top 15 neighborhoods by number of snow emergency tags")
    plt.tight_layout()
    plt.show()


def plot_tags_by_day(df: pd.DataFrame) -> None:
    """Bar plot of tag counts by snow emergency day."""
    if "Day" not in df.columns:
        return

    counts = df["Day"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="darkorange")
    plt.xlabel("Snow emergency day")
    plt.ylabel("Number of tags")
    plt.title("Tags by snow emergency day")
    plt.tight_layout()
    plt.show()


def plot_spatial_scatter(df: pd.DataFrame) -> None:
    """Quick scatter of tag locations using latitude/longitude."""
    if not {"Latitude", "Longitude"} <= set(df.columns):
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(df["Longitude"], df["Latitude"], s=2, alpha=0.4)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial distribution of snow emergency tags")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_data()
    basic_summaries(df)

    # Simple visualizations
    plot_tags_by_neighborhood(df)
    plot_tags_by_day(df)
    plot_spatial_scatter(df)


if __name__ == "__main__":
    main()
