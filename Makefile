
target:
	g++-7 -std=c++17 -O2 -Wall -pedantic main.cpp -pthread

example:
	g++-7 -std=c++17 -O2 -o examples/ex examples/classify_sin_cos.cpp -pthread
