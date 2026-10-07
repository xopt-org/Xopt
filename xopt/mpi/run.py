from xopt.entrypoint.mpi import main


# Legacy support for scripts using old `python -m xopt.mpi.run`
if __name__ == "__main__":
    main()
