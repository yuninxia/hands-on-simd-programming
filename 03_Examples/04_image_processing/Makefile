CXX=g++
CXXFLAGS=-mavx2 -mavx512f -mavx512vl -masm=att -std=c++11
TARGET=simd_program
ASMFILE=main.s
SRCFILE=main.cpp

all: $(TARGET)

$(TARGET): $(SRCFILE)
	$(CXX) $(CXXFLAGS) $(SRCFILE) -o $(TARGET)

asm: $(SRCFILE)
	$(CXX) $(CXXFLAGS) -S $(SRCFILE) -o $(ASMFILE) 

clean:
	rm -f $(TARGET) $(ASMFILE) 