# コンパイラの設定
CC := g++
FLAGS := -std=c++14 -O2

# 訓練プログラムとテストプログラム共通のファイル
COMMON_INC_FILES := $(wildcard common/inc/*.hpp)
COMMON_SRC_FILES := $(wildcard common/src/*.cpp)

# 訓練プログラムとテストプログラム
SRC_TRAIN := src/train.cpp
SRC_TEST := src/test.cpp

# 出力ファイル名
TARGET_TRAIN := out/train.exe
TARGET_TEST := out/test.exe

# プログラムのコンパイル
all: $(TARGET_TRAIN) $(TARGET_TEST)

$(TARGET_TRAIN): $(COMMON_INC_FILES) $(COMMON_SRC_FILES) $(SRC_TRAIN)
	$(CC) $(FLAGS) -o $@ $(SRC_TRAIN) $(COMMON_SRC_FILES)

$(TARGET_TEST): $(COMMON_INC_FILES) $(COMMON_SRC_FILES) $(SRC_TEST)
	$(CC) $(FLAGS) -o $@ $(SRC_TEST) $(COMMON_SRC_FILES)

# プログラムの実行
run: $(TARGET_TRAIN) $(TARGET_TEST)
	@$(TARGET_TRAIN)
	@$(TARGET_TEST)

# プログラムの削除
clean:
	rm -f $(TARGET_TRAIN) $(TARGET_TEST)

.PHONY: all run clean
