#! /usr/bin/zsh


# separate processes b/c TF2 chokes on memory
python3 main.py -o wisconsin.out -m bart -m randomforest -s wisconsin --seed 42
#python3 main.py -o wisconsin.out -m bart -m randomforest -s concrete --seed 42 # OOM error, tf is garbage
#python3 main.py -o wisconsin.out -m bart -m randomforest -s protein --seed 42 # takes long time
python3 main.py -o fires.out -m bart -m randomforest -s fires --seed 42
python3 main.py -o crimes.out -m bart -m randomforest -s crimes --seed 42
python3 main.py -o boston.out -m bart -m randomforest -s boston --seed 42
python3 main.py -o mpg.out -m bart -m randomforest -s mpg --seed 42
