"""Create tables on the Neon database. Idempotent — run as many times as you like."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.db import engine
from app.models import Base


def main():
    print(f"Creating tables on {engine.url.host}/{engine.url.database} ...")
    Base.metadata.create_all(engine)
    print("Done. Tables:")
    for name in Base.metadata.tables:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
