#!/bin/bash
python -m black .
isort .
docformatter -i -r .