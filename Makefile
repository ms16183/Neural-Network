CC = g++
OPTIONS = -std=c++14 -O2

train.exe: train.cpp activation.cpp mnist.cpp error.cpp info.cpp
	$(CC) $(OPTIONS) -o $@ $< activation.cpp mnist.cpp error.cpp info.cpp

run: train.exe
	@./train.exe

clean:
	rm -f ./train.exe
