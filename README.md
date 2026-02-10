
------------------------------------------------------------
INSTALLAZIONE COME PACCHETTO
------------------------------------------------------------

Con miniconda da dentro la cartella del progetto:
    pip install .

------------------------------------------------------------
ESECUZIONE LOCALE
------------------------------------------------------------

EDA:
    python -m src.eda

Training:
    python -m src.training

Predizione:
    python -m src.predict


------------------------------------------------------------
DOCKER
------------------------------------------------------------

Build dell’immagine:
    docker build -t progettoai .

Esecuzione EDA:
    docker run --rm progettoai python -m src.eda

Training (salva il modello in artifacts/):
    docker run --rm -v %cd%/artifacts:/app/artifacts progettoai python -m src.training

Predizione:
    docker run --rm progettoai python -m src.predict

------------------------------------------------------------
CI / CD (GitHub Actions):
 unit, linter, docker build, artifact release
