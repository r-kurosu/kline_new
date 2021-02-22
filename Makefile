a.out : main.o InputData.o
	g++ main.o InputData.o -I/Library/gurobi900/mac64/include -L/Library/gurobi900/mac64/lib -lgurobi_c++ -lgurobi90
main.o : main.cpp InputData.hpp
	g++ -c main.cpp -o main.o -I/Library/gurobi900/mac64/include -L/Library/gurobi900/mac64/lib -lgurobi_c++ -lgurobi90
InputData.o : InputData.cpp InputData.hpp 
	g++ -c InputData.cpp -o InputData.o