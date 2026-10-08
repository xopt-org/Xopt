"""Generator implementations that run computation on a remote HTTP server.

Importing this package does not pull in ``requests``/``fastapi``/``uvicorn`` -
submodules are imported lazily by ``xopt.generators.get_generator_dynamic``.
"""
