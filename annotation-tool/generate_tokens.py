"""Generate access tokens for annotators.

Usage:
  python generate_tokens.py

Reads annotators from the database, generates a unique access token for each,
stores the bcrypt hash in the DB, and prints the plaintext tokens to stdout.
Share each token privately with the corresponding annotator.
"""

import secrets
import bcrypt
from app.database import engine, SessionLocal, Base
from app.models import Annotator


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        annotators = db.query(Annotator).all()
        if not annotators:
            print("No annotators found in database. Load data first.")
            return

        print("Generated access tokens (share privately with each annotator):\n")
        print("-" * 50)
        for ann in annotators:
            token = secrets.token_urlsafe(16)
            hashed = bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            ann.access_token_hash = hashed
            print(f"  {ann.name}: {token}")
        print("-" * 50)
        db.commit()
        print("\nTokens saved to database.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
