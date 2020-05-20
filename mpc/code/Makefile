OS := $(shell uname)
ifeq ($(OS),Darwin)
  # Run MacOS commands
  CPP = /usr/bin/clang++
  INCPATHS = -I~/projects/local/include
else
  # check for Linux and run other commands
  CPP = g++
  INCPATHS = -I/usr/bin/include -Iinclude
endif

CFLAGS = -g -Wall -O2 $(INCPATHS) -march=native -std=c++11 -pthread -Wno-sign-compare
LDLIBS = -lssl -lcrypto -lntl -lgmp -lm -ldl -lboost_system -lboost_filesystem
LDPATH = -L/usr/local/lib

BUILD = build
PROGS = bin

SRC = crypto.cpp param.cpp connect.cpp aesstream.cpp mpc_int.cpp
#PROGNAMES = ShareData TrainSecureDTI GenerateKey AnalyzeFeature test UnitTestClient
#PROGNAMES = UnitTestClient ShareDataECG
PROGNAMES = TrainSecureECG ShareDataECG UnitTestClient
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
