An integration test can touch external systems (File IO, Network IO, Database, External Web Services...)
RUN WITH:
pytest -m "not integtest"