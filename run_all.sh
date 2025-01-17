#! /usr/bin/zsh


# separate processes b/c TF2 chokes on memory
for i in {0..32}
do
	python3 main.py -o wisconsin_$i.out -m bart -m randomforest -s wisconsin --seed $i --strength01 10000 --strength12 1000
	python3 main.py -o concrete_$i.out -m bart -m randomforest -s concrete --seed $i -d 5
	#python3 main.py -o protein_$i.out -m bart -m randomforest -s protein --seed $i
	python3 main.py -o fires_$i.out -m bart -m randomforest -s fires --seed $i
	python3 main.py -o crimes_$i.out -m bart -m randomforest -s crimes --seed $i -d 5 --strength01 10000 --strength12 10000
	python3 main.py -o boston_$i.out -m bart -m randomforest -s boston --seed $i --strength01 10000 --strength12 10000
	python3 main.py -o mpg_$i.out -m bart -m randomforest -s mpg --seed $i
done


python3 tablize.py -o res_all.csv -f wisconsin -f concrete -f fires -f crimes -f boston -f mpg
