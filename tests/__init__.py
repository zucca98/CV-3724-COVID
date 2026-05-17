"""Package `tests` - test suite pytest del progetto.

Filosofia di testing:
- Almeno un *smoke test* per ogni modulo in `src/`.
- Test "ermetici": niente download, niente dataset reale, niente GPU.
  Uso esclusivo di immagini sintetiche generate al volo con NumPy + PIL.
- `tmp_path` di pytest per creare file temporanei isolati per test.
- Seed fissi (`default_rng(42)`) ovunque per riproducibilita'.

Lancia tutti i test con:
    pytest tests/ -v
"""
