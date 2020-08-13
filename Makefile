CC = g++
OPTIONS = -std=c++14 -O2

train.exe: train.cpp
	$(CC) $(OPTIONS) -o $@ $<

run: train.exe
	@./train.exe

clean:
	rm -f ./train.exe
