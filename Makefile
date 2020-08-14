CC = g++
OPTIONS = -std=c++14 -O2

train.exe: train.cpp activation.cpp
	$(CC) $(OPTIONS) -o $@ $< activation.cpp

run: train.exe
	@./train.exe

clean:
	rm -f ./train.exe
