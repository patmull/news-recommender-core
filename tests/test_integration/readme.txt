An integration test can touch external systems (File IO, Network IO, Database, External Web Services...)
FOR LEAVING INTEGTESTS OUT, RUN WITH:
pytest -m "not integtest"
