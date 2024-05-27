.PHONY: help
help:
	@echo "Available Targets"
	@echo "Usage: "
	@echo " make            : Compile wrapper and produce executable spvwrapper.exe (Default)"
	@echo " make clean      : Remove compiled files and executable (exe)"
	@echo " make VERBOSE=1  : Enable application to produce logs"
	@echo " make DEBUG=1    : Enable application to with debug info"
	@echo " make help       : Display this information "

CXX = icpx
CXXFLAGS = -std=c++20 -fsycl
LDFLAGS = -lpthread -lm -ldl -lze_loader
SPVEXE = tritonspvc++.exe
.DEFAULT_GOAL = $(SPVEXE)
VERBOSE ?= 0
ifeq ($(VERBOSE),1)
	CXXFLAGS += -DVERBOSE
endif

DEBUG ?= 0
ifeq ($(DEBUG),1)
	CXXFLAGS += -g
endif

all: $(SPVEXE) $(MMEXE)

SRC = $(PWD)/src

$(SPVEXE): $(SRC)/wrapper_sycl.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f *.exe 
