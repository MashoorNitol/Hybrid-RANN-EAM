
SRC_DIR := ./
OBJ_DIR := ./obj
#TEST_DIR := ../calibration_testing/TiZr_fit/fit_1
#TEST_FILE := ZrTi1.nn
TEST_DIR := /mnt/wwn-0x5000c500a2b12ba1-part2/cdb333/Altraining
TEST_FILE := MgAl_900.nn
#TEST_DIR := ../calibration_testing/Mg_pure
#TEST_FILE := Mg_new.nn
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
REPFLAGS = -qopt-report=5 -qopt-report-filter="calibration.cpp,1802-2101"
DEPENDS := ${OBJ_FILES:.o=.d}  # substitutes ".o" with ".d"
# https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html
# MMD: to handle dependency by includ header locations in .d file

## for debugging using gnu compiler
# TARGET := nn_calib_gcc_debug
# CXX := g++
# CXXFLAGS := -ggdb -fsanitize=address -fno-omit-frame-pointer -lasan -fopenmp
# LDFLAGS := -lasan -fopenmp

## for profiling using gnu compiler
#TARGET := nn_calib_gcc
#CXX := g++
#CXXFLAGS := -pg -O2 -shared-libgcc -Wall -MMD -fopenmp
#LDFLAGS := -fopenmp

## for profiling using intel compiler
#TARGET := nn_calib_icc
#CXX := icpc
#CXXFLAGS := -g -O2 -xHost -shared-intel -debug inline-debug-info -D TBB_USE_THREADING_TOOLS -qopenmp -qopenmp-link dynamic -parallel-source-info=2 -Wall -MMD
#LDFLAGS := -qopenmp

## for production run using intel compiler
#TARGET := nn_calib_icc_new
#CXX := icpc
#CXXFLAGS := -xHost -O2 -fp-model fast=2 -no-prec-div -qoverride-limits -mavx -qopenmp
#LDFLAGS := -qopenmp

TARGET := nn_calib_icc_new
CXX := icpc
CXXFLAGS := -xHost -O2 -fp-model fast=2 -no-prec-div -qoverride-limits -mavx -qopenmp -std=c++11
LDFLAGS := -qopenmp


## for production run using gcc compiler
#TARGET := nn_calib_gcc_new
#CXX := g++
#CXXFLAGS := -O2 -shared-libgcc -MMD -fopenmp
#LDFLAGS := -fopenmp


LDLIBS :=


all: $(TARGET)

# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
# $@: the file name of the target. here nn_calibration
# $^: the names of all the prerequisites, with spaces between them. here $(OBJ_FILES)
# $<: the name of the first prerequisite, e.g. activation.cpp
$(TARGET): $(OBJ_FILES)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
	cp $(TARGET) $(TEST_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run:
	$(info )
	$(info Run the default test case on CPU: )
	(cd $(TEST_DIR) && ./$(TARGET) -in $(TEST_FILE))

clean:
	-rm -rf $(OBJ_DIR) $(TARGET)

-include ${DEPENDS}           # copies files x.d, y.d, z.d (if they exist)

#----------------------------------------------------------------
#---------- Intel Advisor Analysis ------------------------------
#----------------------------------------------------------------

ADVPRJ := $(CURDIR)/intel_advisor

survey:
	(cd $(TEST_DIR) && advixe-cl -collect survey -project-dir $(ADVPRJ) -- ./$(TARGET) -in $(TEST_FILE) )

roofline:
	(cd $(TEST_DIR) && advixe-cl -collect survey -project-dir $(ADVPRJ) -- ./$(TARGET) -in $(TEST_FILE))
	(cd $(TEST_DIR) && advixe-cl -collect tripcounts -flop -project-dir $(ADVPRJ) -- ./$(TARGET) -in $(TEST_FILE))

map:
	(cd $(TEST_DIR) && advixe-cl -collect map -mark-up-list=1 -project-dir $(ADVPRJ) -- ./$(TARGET) -in $(TEST_FILE))

open-gui:
	advixe-gui $(ADVPRJ)/$(ADVPRJ).advixeproj >/dev/null 2>&1 &

clean-results:
	rm -rf $(ADVPRJ)
