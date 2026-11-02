"""The startup migration adds columns introduced after a database was first created."""

from sqlalchemy import create_engine, inspect, text

from database.connection import _add_missing_columns


def test_old_predictions_table_gains_new_columns(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE predictions (id INTEGER PRIMARY KEY, prediction_id VARCHAR(50), final_condition VARCHAR(100))"
        ))
        conn.execute(text("INSERT INTO predictions (prediction_id, final_condition) VALUES ('p1', 'Normal')"))

    with eng.begin() as conn:
        _add_missing_columns(conn)
        _add_missing_columns(conn)  # idempotent

    cols = {c["name"] for c in inspect(eng).get_columns("predictions")}
    assert {"probabilities", "model_sha256", "inconclusive"} <= cols
    with eng.connect() as conn:
        assert conn.execute(text("SELECT final_condition, probabilities FROM predictions")).one() == ("Normal", None)
