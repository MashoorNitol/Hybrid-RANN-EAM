/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */


#include "pair_rann.h"

using namespace LAMMPS_NS;

//read command line input.
int main(int argc, char **argv)
{
	char str[MAXLINE];
	if (argc!=3 || strcmp(argv[1],"-in")!=0){
		sprintf(str,"syntax: nn_calibration -in \"input_file.rann\"\n");
		std::cout<<str;
	}
	else{
		PairRANN *cal = new PairRANN(argv[2]);
		cal->setup();
		cal->run();
		cal->finish();
		delete cal;
	}
}

