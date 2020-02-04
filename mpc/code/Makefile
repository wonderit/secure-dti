OS := $(shell uname)
ifeq ($(OS),Darwin)
  # Run MacOS commands
  CPP = /usr/bin/clang++
  INCPATHS = -I/Users/wonsuk/projects/local/include
else
  # check for Linux and run other commands
  CPP = g++
  INCPATHS = -I/usr/bin/include -Iinclude
endif

CFLAGS = -g -Wall -O1 $(INCPATHS) -march=native -std=c++11 -pthread -Wno-sign-compare
LDLIBS = -lssl -lcrypto -lntl -lgmp -lm -ldl
LDPATH = -L/usr/local/lib

BUILD = build
PROGS = bin

SRC = crypto.cpp connect.cpp aesstream.cpp mpc.cpp param.cpp
#PROGNAMES = ShareData TrainSecureDTI GenerateKey AnalyzeFeature test UnitTestClient
PROGNAMES = TrainSecureECG ShareDataECG GenerateKey
#PROGNAMES = TrainSecureMNIST ShareDataMNIST

OBJPATHS = $(patsubst %.cpp,$(BUILD)/%.o, $(SRC))
TESTPATHS = $(addprefix $(PROGS)/, $(PROGNAMES))

all: $(OBJPATHS) $(TESTPATHS)

obj: $(OBJPATHS)

$(BUILD):
	mkdir -p $(BUILD)

$(PROGS):
	mkdir -p $(PROGS)

$(BUILD)/%.o: %.cpp *.h | $(BUILD)
	$(CPP) $(CFLAGS) -o $@ -c $<

$(PROGS)/%: %.cpp $(OBJPATHS) $(PROGS)
	$(CPP) $(CFLAGS) -o $@ $< $(LDPATH) $(OBJPATHS) $(LDLIBS)

clean:
	rm -rf $(OBJPATHS) $(TESTPATHS) *~
