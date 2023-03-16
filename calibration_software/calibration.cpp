#include "omp.h"
#include "pair_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"


using namespace LAMMPS_NS;

PairRANN::PairRANN(char *potential_file){
	cutmax = 0.0;
	nelementsp = -1;
	nelements = -1;
	net = NULL;
	fingerprintlength = NULL;
	mass = NULL;
	betalen = 0;
	doregularizer = false;
	normalizeinput = true;
	fingerprints = NULL;
	max_epochs = 1e7;
	regularizer = 0.0;
	res = 10000;
	fingerprintcount = 0;
	stateequationcount = 0;
	elementsp = NULL;
	elements = NULL;
	activation = NULL;
	state = NULL;
	tolerance = 1e-6;
	sims = NULL;
	doscreen = false;
	allscreen = true;
	dospin = false;
	map = NULL;//check this
	natoms = 0;
	nsims = 0;
	doforces = false;
	fingerprintperelement = NULL;
	stateequationperelement = NULL;
	validation = 0.0;
	potential_output_freq = 100;
	algorithm = new char [SHORTLINE];
	potential_input_file = new char [strlen(potential_file)+1];
	dump_directory = new char [SHORTLINE];
	log_file = new char [SHORTLINE];
	potential_output_file = new char [SHORTLINE];
	strncpy(this->potential_input_file,potential_file,strlen(potential_file)+1);
	char temp1[] = ".\0";
	char temp2[] = "calibration.log\0";
	char temp3[] = "potential_output.nn\0";
	strncpy(dump_directory,temp1,strlen(temp1)+1);
	strncpy(log_file,temp2,strlen(temp2)+1);
	strncpy(potential_output_file,temp3,strlen(temp3)+1);
	strncpy(algorithm,"LM_ch",strlen("LM_ch")+1);
	overwritepotentials = false;
	debug_level1_freq = 10;
	debug_level2_freq = 0;
	debug_level3_freq = 0;
	debug_level4_freq = 0;
	debug_level5_freq = 0;
	debug_level6_freq = 0;
	adaptive_regularizer = false;
	seed = time(0);
	lambda_initial = 1000;
	lambda_increase = 10;
	lambda_reduce = 0.3;
}

PairRANN::~PairRANN(){
	//clear memory
	delete [] algorithm;
	delete [] potential_input_file;
	delete [] dump_directory;
	delete [] log_file;
	delete [] potential_output_file;
	delete [] r;
	delete [] v;
	delete [] Xset;
	delete [] mass;
	for (int i=0;i<nsims;i++){
		for (int j=0;j<sims[i].inum;j++){
//			delete [] sims[i].x[j];
			if (doforces)delete [] sims[i].f[j];
			if (sims[i].spins)delete [] sims[i].s[j];
			delete [] sims[i].firstneigh[j];
			delete [] sims[i].features[j];
			if (doforces)delete [] sims[i].dfx[j];
			if (doforces)delete [] sims[i].dfy[j];
			if (doforces)delete [] sims[i].dfz[j];
		}
//		delete [] sims[i].x;
		if (doforces)delete [] sims[i].f;
		if (sims[i].spins)delete [] sims[i].s;
		if (doforces)delete [] sims[i].dfx;
		if (doforces)delete [] sims[i].dfy;
		if (doforces)delete [] sims[i].dfz;
		delete [] sims[i].firstneigh;
		delete [] sims[i].id;
		delete [] sims[i].features;
		delete [] sims[i].ilist;
		delete [] sims[i].numneigh;
		delete [] sims[i].type;
	}
	delete [] sims;
	for (int i=0;i<nelements;i++){delete [] elements[i];}
	delete [] elements;
	for (int i=0;i<nelementsp;i++){delete [] elementsp[i];}
	delete [] elementsp;
	for (int i=0;i<=nelements;i++){
		if (net[i].layers>0){
			for (int j=0;j<net[i].layers-1;j++){
				delete activation[i][j];
				delete [] net[i].bundleinputsize[j];
				delete [] net[i].bundleoutputsize[j];
				for (int k=0;k<net[i].bundles[j];k++){
					delete [] net[i].bundleinput[j][k];
					delete [] net[i].bundleoutput[j][k];
					delete [] net[i].bundleW[j][k];
					delete [] net[i].bundleB[j][k];
					delete [] net[i].freezeW[j][k];
					delete [] net[i].freezeB[j][k];
				}
				delete [] net[i].bundleinput[j];
				delete [] net[i].bundleoutput[j];
				delete [] net[i].bundleW[j];
				delete [] net[i].bundleB[j];
				delete [] net[i].freezeW[j];
				delete [] net[i].freezeB[j];
			}
			delete [] activation[i];
			delete [] net[i].dimensions;
			delete [] net[i].startI;
			delete [] net[i].bundleinput;
			delete [] net[i].bundleoutput;
			delete [] net[i].bundleinputsize;
			delete [] net[i].bundleoutputsize;
			delete [] net[i].bundleW;
			delete [] net[i].bundleB;
			delete [] net[i].freezeW;
			delete [] net[i].freezeB;
			delete [] net[i].bundles;
		}
	}
	delete [] net;
	delete [] map;
	for (int i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			for (int j=0;j<fingerprintperelement[i];j++){
				delete fingerprints[i][j];
			}
			delete [] fingerprints[i];
		}
		if (stateequationperelement[i]>0){
			for (int j=0;j<stateequationperelement[i];j++){
				delete state[i][j];
			}
			delete [] state[i];
		}
	}
	delete [] fingerprints;
	delete [] activation;
	delete [] state;
	delete [] fingerprintcount;
	delete [] fingerprintperelement;
	delete [] fingerprintlength;
	delete [] stateequationcount;
	delete [] stateequationperelement;
	delete [] freezebeta;
}

void PairRANN::setup(){

	int nthreads=1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();

	std::cout << std::endl;
	std::cout << "# Number Threads     : " << nthreads << std::endl;

	double start_time = omp_get_wtime();

	read_file(potential_input_file);
	check_parameters();
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<fingerprintperelement[i];j++){
			  fingerprints[i][j]->allocate();
		}
		for (int j=0;j<stateequationperelement[i];j++){
			state[i][j]->allocate();
		}
	}
	read_dump_files();
	create_neighbor_lists();
	compute_fingerprints();
	if (normalizeinput){
		normalize_data();
	}
	separate_validation();

	double end_time = omp_get_wtime();
	double time = (end_time-start_time);
	printf("finished setup(): %f seconds\n",time);
}

void PairRANN::run(){
	if (strcmp(algorithm,"LMqr")==0){
		//DEPRECATED. Do not use.
		//slow but robust.
		//levenburg_marquardt_qr();
		errorf(FLERR,"QR algorithm is discontinued.");
	}
	else if (strcmp(algorithm,"LMch")==0){
		//faster. crashes if Jacobian has any columns of zeros.
		//usually will find exactly the same step for each iteration as qr.
		levenburg_marquardt_ch();
	}
	else if (strcmp(algorithm,"CG")==0){
		//NYI
		//faster iterations, but less accurate steps.
		//best parallelization

	}
	else {
		errorf("unrecognized algorithm");
	}
}


void PairRANN::finish(){

//	write_potential_file(true);
}

void PairRANN::read_parameters(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
	if (line[1]=="algorithm"){
		if (line[1].size()>SHORTLINE){
			delete [] algorithm;
			algorithm = new char[line[1].size()+1];
		}
		strncpy(algorithm,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="dumpdirectory"){
		if (line1[0].size()>SHORTLINE){
			delete [] dump_directory;
			dump_directory = new char[line1[0].size()+1];
		}
		strncpy(dump_directory,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="doforces"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		doforces = (temp>0);
	}
	else if (line[1]=="normalizeinput"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		normalizeinput = (temp>0);
	}
	else if (line[1]=="tolerance"){
		tolerance = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="regularizer"){
		regularizer = strtod(line1[0].c_str(),NULL);
		if (regularizer!=0.0){doregularizer = true;}
	}
	else if (line[1]=="logfile"){
		if (line1[0].size()>SHORTLINE){
			delete [] log_file;
			log_file = new char [line1[0].size()+1];
		}
		strncpy(log_file,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="potentialoutputfreq"){
		potential_output_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="potentialoutputfile"){
		if (line1[0].size()>SHORTLINE){
			delete [] potential_output_file;
			potential_output_file = new char [line1[0].size()+1];
		}
		strncpy(potential_output_file,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="maxepochs"){
		max_epochs = strtol(line1[0].c_str(),NULL,10);
	}
	// else if (line[1]=="dimsreserved"){
	// 	int i;
	// 	for (i=0;i<nelements;i++){
	// 		if (strcmp(words[2],elements[i])==0){
	// 			if (net[i].layers==0)errorf("networklayers for each atom type must be defined before the corresponding layer sizes.");
	// 			int j = strtol(words[3],NULL,10);
	// 			net[i].dimensionsr[j]= strtol(line1,NULL,10);
	// 			return;
	// 		}
	// 	}
	// 	errorf("dimsreserved element not found in atom types");
	// }
	else if (line[1]=="freezeW"){
		int i,j,k,b,l,ins,ops;
		char **words1,*ptr;
		char linetemp [MAXLINE];
		int nwords = line.size();
		if (nwords == 4){b=0;}
		else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
		for (l=0;l<nelements;l++){
			if (line[2]==elements[l]){
				if (net[l].layers==0)errorf("networklayers must be defined before weights.");
				i=strtol(line[3].c_str(),NULL,10);
				if (i>=net[l].layers || i<0)errorf("invalid weight layer");
				if (dimensiondefined[l][i]==false || dimensiondefined[l][i+1]==false) errorf("network layer sizes must be defined before corresponding weight");
				if (bundle_inputdefined[l][i][b]==false && b!=0) errorf("bundle inputs must be defined before weights");
				if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before weights");
				if (net[l].identitybundle[i][b]) errorf("cannot define weights for an identity bundle");
				if (bundle_inputdefined[l][i][b]==false){ins = net[l].dimensions[i];}
				else {ins = net[l].bundleinputsize[i][b];}
				if (bundle_outputdefined[l][i][b]==false){ops = net[l].dimensions[i+1];}
				else {ops = net[l].bundleoutputsize[i][b];}
				net[l].freezeW[i][b] = new bool [ins*ops];
				nwords = line1.size();
				if (nwords != ins)errorf("invalid weights per line");
				for (k=0;k<ins;k++){
					net[l].freezeW[i][b][k] = strtol(line1[k].c_str(),NULL,10);
				}
				for (j=1;j<ops;j++){
					ptr = fgets(linetemp,MAXLINE,fp);
					(linenum)++;
					line1 = tokenmaker(linetemp,": ,\t_\n");
					if (ptr==NULL)errorf("unexpected end of potential file!");
					nwords = line1.size();
					if (nwords != ins)errorf("invalid weights per line");
					for (k=0;k<ins;k++){
						net[l].freezeW[i][b][j*ins+k] = strtol(line1[k].c_str(),NULL,10);
					}
				}
				delete [] words1;
				return;
			}
		}
		errorf("weight element not found in atom types");
	}
	else if (line[1]=="freezeB"){
		int i,j,l,b,ops;
		char *ptr;
		int nwords = line.size();
		char linetemp[MAXLINE];
		if (nwords == 4){b=0;}
		else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
		for (l=0;l<nelements;l++){
			if (line[2]==elements[l]){
				if (net[l].layers==0)errorf("networklayers must be defined before biases.");
				i=strtol(line[3].c_str(),NULL,10);
				if (i>=net[l].layers || i<0)errorf("invalid bias layer");
				if (dimensiondefined[l][i]==false) errorf("network layer sizes must be defined before corresponding bias");
				if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before bias");
				if (net[l].identitybundle[i][b]) errorf("cannot define bias for an identity bundle");
				if (bundle_outputdefined[l][i][b]==false){ops=net[l].dimensions[i+1];}
				else {ops = net[l].bundleoutputsize[i][b];}
				net[l].freezeB[i][b] = new bool [ops];
				net[l].freezeB[i][b][0] = strtol(line1[0].c_str(),NULL,10);
				for (j=1;j<ops;j++){
					ptr = fgets(linetemp,MAXLINE,fp);
					if (ptr==NULL)errorf("unexpected end of potential file!");
					line1 = tokenmaker(linetemp," ,\t:_\n");
					net[l].freezeB[i][b][j] = strtol(line1[0].c_str(),NULL,10);
				}
				return;
			}
		}
		errorf("bias element not found in atom types");
	}
	else if (line[1]=="validation"){
		validation = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="overwritepotentials") {
		int temp = strtol(line1[0].c_str(),NULL,10);
		overwritepotentials = (temp>0);
	}
	else if (line[1]=="debug1freq") {
		debug_level1_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug2freq") {
		debug_level2_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug3freq") {
		debug_level3_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug4freq") {
		debug_level4_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug5freq") {
		debug_level5_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug6freq") {
		debug_level6_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="adaptiveregularizer") {
		double temp = strtod(line1[0].c_str(),NULL);
		adaptive_regularizer = (temp>0);
		doregularizer = true;
	}
	else if (line[1]=="lambdainitial"){
		lambda_initial = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdaincrease"){
		lambda_increase = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdareduce"){
		lambda_reduce = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="seed"){
		seed = strtol(line1[0].c_str(),NULL,10);
	}
	else {
		char str[MAXLINE];
		sprintf(str,"unrecognized keyword in parameter file: %s\n",line[1]);
		errorf(filename,*linenum,str);
	}
}

void PairRANN::create_random_weights(int rows,int columns,int itype,int layer,int bundle){
	net[itype].bundleW[layer][bundle] = new double [rows*columns];
	net[itype].freezeW[layer][bundle] = new bool [rows*columns];
	double r;
	for (int i=0;i<rows;i++){
		for (int j=0;j<columns;j++){
			r = (double)rand()/RAND_MAX*2-1;//flat distribution from -1 to 1
			net[itype].bundleW[layer][bundle][i*columns+j] = r;
			net[itype].freezeW[layer][bundle][i*columns+j] = 0;
		}
	}
	weightdefined[itype][layer][bundle]=true;
}

void PairRANN::create_random_biases(int rows,int itype, int layer,int bundle){
	net[itype].bundleB[layer][bundle] = new double [rows];
	net[itype].freezeB[layer][bundle] = new bool [rows];
	double r;
	for (int i=0;i<rows;i++){
		r = (double) rand()/RAND_MAX*2-1;
		net[itype].bundleB[layer][bundle][i] = r;
		net[itype].freezeB[layer][bundle][i] = 0;
	}
	biasdefined[itype][layer][bundle]=true;
}

void PairRANN::allocate(const std::vector<std::string> &elementwords)
{
	int i,n;
	cutmax = 0;
	nelementsp=nelements+1;
	//initialize arrays
	elements = new char *[nelements];
	elementsp = new char *[nelementsp];//elements + 'all'
	map = new int[nelementsp];
	mass = new double[nelements];
	net = new NNarchitecture[nelementsp];
	for (i=0;i<nelementsp;i++){net[i].layers=0;}
	betalen_v = new int[nelementsp];
	betalen_f = new int[nelementsp];
	screening_min = new double [nelements*nelements*nelements];
	screening_max = new double [nelements*nelements*nelements];
	for (i=0;i<nelements;i++){
		for (int j =0;j<nelements;j++){
			for (int k=0;k<nelements;k++){
				screening_min[i*nelements*nelements+j*nelements+k] = 0.8;//default values. Custom values may be read from potential file later.
				screening_max[i*nelements*nelements+j*nelements+k] = 2.8;//default values. Custom values may be read from potential file later.
			}
		}
	}
	weightdefined = new bool**[nelementsp];
	biasdefined = new bool **[nelementsp];
	dimensiondefined = new bool*[nelements];
	bundle_inputdefined = new bool**[nelements];
	bundle_outputdefined = new bool**[nelements];
	activation = new RANN::Activation***[nelementsp];
	fingerprints = new RANN::Fingerprint**[nelementsp];
	state = new RANN::State**[nelementsp];
	fingerprintlength = new int[nelementsp];
	fingerprintperelement = new int [nelementsp];
	fingerprintcount = new int[nelementsp];
	stateequationperelement = new int [nelementsp];
	stateequationcount = new int [nelementsp];
	for (i=0;i<=nelements;i++){
		n = elementwords[i].size();
		fingerprintlength[i]=0;
		fingerprintperelement[i] = -1;
		fingerprintcount[i] = 0;
		stateequationperelement[i] = 0;
		stateequationcount[i] = 0;
		map[i] = i;
		if (i<nelements){
			mass[i]=-1.0;
			elements[i]= utils::strdup(elementwords[i]);
		}
		elementsp[i]= utils::strdup(elementwords[i]);
	}

}


void PairRANN::update_stack_size(){
	//TO DO: fix. Still getting stack overflow from underestimating memory needs.
	//get very rough guess of memory usage
	int jlen = nsims;
	if (doregularizer){
		jlen+=betalen-1;
	}
	if (doforces){
		jlen+=natoms*3;
	}
	//neighborlist memory use:
	memguess = 0;
	for (int i=0;i<nelementsp;i++){
		memguess+=8*net[i].dimensions[0]*20*3;
	}
	memguess+=8*20*12;
	memguess+=8*20*20*3;
	//separate validation memory use:
	memguess+=nsims*8*2;
	//levenburg marquardt ch memory use:
	memguess+=8*jlen*betalen*2;
	memguess+=8*betalen*betalen;
	memguess+=8*jlen*4;
	memguess+=8*betalen*4;
	//chsolve memory use:
	memguess+=8*betalen*betalen;
	//generous buffer:
	memguess *= 16;
	const rlim_t kStackSize = memguess;
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur += kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

bool PairRANN::check_parameters(){
	int itype,layer,bundle,rows,columns,r,c,count;
	if (strcmp(algorithm,"LMqr")!=0 && strcmp(algorithm,"LMch")!=0)errorf(FLERR,"Unrecognized algorithm. Must be LM_ch or LM_qr\n");//add others later maybe
	if (tolerance==0.0)errorf("tolerance not correctly initialized\n");
	if (tolerance<0.0 || max_epochs < 0 || regularizer < 0.0 || potential_output_freq < 0)errorf("detected parameter with negative value which must be positive.\n");
	srand(seed);
	count=0;
	//populate vector of frozen parameters
	betalen = 0;
	for (itype=0;itype<nelementsp;itype++){
		for (layer=0;layer<net[itype].layers-1;layer++){
			for (bundle=0;bundle<net[itype].bundles[layer];bundle++){
				if (net[itype].identitybundle[layer][bundle]){continue;}
				rows = net[itype].bundleoutputsize[layer][bundle];
				columns = net[itype].bundleinputsize[layer][bundle];
				betalen += rows*columns+rows;
			}
		}
	}
	freezebeta = new bool[betalen];
	for (itype=0;itype<nelementsp;itype++){
		for (layer=0;layer<net[itype].layers-1;layer++){
			for (bundle=0;bundle<net[itype].bundles[layer];bundle++){
				if (net[itype].identitybundle[layer][bundle]){continue;}
				rows = net[itype].bundleoutputsize[layer][bundle];
				columns = net[itype].bundleinputsize[layer][bundle];
				for (r=0;r<rows;r++){
					for (c=0;c<columns;c++){
						if (net[itype].freezeW[layer][bundle][r*columns+c]){
							freezebeta[count] = 1;
							count++;
						}
						else {
							freezebeta[count] = 0;
							count++;
						}
					}
					if (net[itype].freezeB[layer][bundle][r]){
						freezebeta[count] = 1;
						count++;
					}
					else {
						freezebeta[count] = 0;
						count++;
					}
				}
			}
		}
		betalen_v[itype]=count;
	}
	betalen = count;//update betalen to skip frozen parameters

	return false;//everything looks good
}

//part of setup. Do not optimize:
void PairRANN::read_dump_files(){
	DIR *folder;
//	char str[MAXLINE];
	struct dirent *entry;
	int file = 0;
	char line[MAXLINE],*ptr;
	char **words;
	int nwords,nwords1,sets;
	folder = opendir(dump_directory);

	if(folder == NULL)
	{
		errorf("unable to open dump directory");
	}
	std::cout<<"reading dump files\n";
	int nsims = 0;
	int nsets = 0;
	//count files
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		nsets++;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	this->nsets = nsets;
	Xset=new int[nsets];
	dumpfilenames = new char*[nsets];
	int count=0;
	//count snapshots per file
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		dumpfilenames[count] = new char[strlen(entry->d_name)+10];
		strcpy(dumpfilenames[count],entry->d_name);
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		ptr = fgets(line,MAXLINE,fid);
		nwords = 0;
		words = new char* [strlen(line)];
		words[nwords++] = strtok(line," ,\t\n");
		while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
		nwords--;
		if (nwords!=5 && nwords != 11){errorf("dumpfile must contain 2nd line with timestep, energy, energy_weight, force_weight, snapshots\n");}
		sets = strtol(words[4],NULL,10);
		delete [] words;
		nsims+=sets;
		Xset[count++]=sets;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	sims = new Simulation[nsims];
	this->nsims = nsims;
	sims[0].startI=0;
	//read dump files
	while((entry=readdir(folder))){
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		printf("\t%s\n",entry->d_name);
		if (!fid){continue;}
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		while (ptr!=NULL){
			if (strstr(line,"ITEM: TIMESTEP")==NULL)errorf("invalid dump file line 1");
			ptr = fgets(line,MAXLINE,fid);//timestep
			nwords = 0;
			char *words1[strlen(line)];
			words1[nwords++] = strtok(line," ,\t");
			while ((words1[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			if (nwords!=5 && nwords != 11)errorf("error: dump file line 2 must contain 5 entries: timestep, energy, energy_weight, force_weight, snapshots");
			int timestep = strtol(words1[0],NULL,10);
			sims[file].filename = new char [strlen(entry->d_name)+10];
			sims[file].timestep = timestep;
			strcpy(sims[file].filename,entry->d_name);
			sims[file].energy = strtod(words1[1],NULL);
			sims[file].energy_weight = strtod(words1[2],NULL);
			sims[file].force_weight = strtod(words1[3],NULL);
			sims[file].spinvec[0] = 0;
			sims[file].spinvec[1] = 0;
			sims[file].spinvec[2] = 0;
			sims[file].spinaxis[0] = 0;
			sims[file].spinaxis[1] = 0;
			sims[file].spinaxis[2] = 0;
			if (nwords==11){
				sims[file].spinspirals = true;
				double spinvec[3],spinaxis[3];
				spinvec[0] = strtod(words1[5],NULL);
				spinvec[1] = strtod(words1[6],NULL);
				spinvec[2] = strtod(words1[7],NULL);
				spinaxis[0] = strtod(words1[8],NULL);
				spinaxis[1] = strtod(words1[9],NULL);
				spinaxis[2] = strtod(words1[10],NULL);
				double norm = spinaxis[0]*spinaxis[0]+spinaxis[1]*spinaxis[1]+spinaxis[2]*spinaxis[2];
				if (norm<1e-14){errorf("spinaxis cannot be zero\n");}
				spinaxis[0]=spinaxis[0]/sqrt(norm);
				spinaxis[1]=spinaxis[1]/sqrt(norm);
				spinaxis[2]=spinaxis[2]/sqrt(norm);
				sims[file].spinvec[0] = spinvec[0];
				sims[file].spinvec[1] = spinvec[1];
				sims[file].spinvec[2] = spinvec[2];
				sims[file].spinaxis[0] = spinaxis[0];
				sims[file].spinaxis[1] = spinaxis[1];
				sims[file].spinaxis[2] = spinaxis[2];
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: NUMBER OF ATOMS
			if (strstr(line,"ITEM: NUMBER OF ATOMS")==NULL)errorf("invalid dump file line 3");
			ptr = fgets(line,MAXLINE,fid);//natoms
			int natoms = strtol(line,NULL,10);
			//printf("%d %d %f\n",file,timestep,sims[file].energy/natoms);
			if (file>0){sims[file].startI=sims[file-1].startI+natoms*3;}
			this->natoms+=natoms;
			sims[file].energy_weight /=natoms;
			//sims[file].force_weight /=natoms;
			ptr = fgets(line,MAXLINE,fid);//ITEM: BOX BOUNDS xy xz yz pp pp pp
			if (strstr(line,"ITEM: BOX BOUNDS")==NULL)errorf("invalid dump file line 5");
			double box[3][3];
			double origin[3];
			bool cols[12];
			for (int i= 0;i<11;i++){
				cols[i]=false;
			}
			box[0][1] = box[0][2] = box[1][2] = 0.0;
			for (int i = 0;i<3;i++){
				ptr = fgets(line,MAXLINE,fid);//box line
				char *words[4];
				nwords = 0;
				words[nwords++] = strtok(line," ,\t\n");
				while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
				nwords--;
				if (nwords!=3 && nwords!=2){errorf("invalid dump box definition");}
				origin[i] = strtod(words[0],NULL);
				box[i][i] = strtod(words[1],NULL);
				if (nwords==3){
					if (i==0){
						box[0][1]=strtod(words[2],NULL);
						if (box[0][1]>0){box[0][0]-=box[0][1];}
						else origin[0] -= box[0][1];
					}
					else if (i==1){
						box[0][2]=strtod(words[2],NULL);
						if (box[0][2]>0){box[0][0]-=box[0][2];}
						else origin[0] -= box[0][2];
					}
					else{
						box[1][2]=strtod(words[2],NULL);
						if (box[1][2]>0)box[1][1]-=box[1][2];
						else origin[1] -=box[1][2];
					}
				}
			}
			for (int i=0;i<3;i++)box[i][i]-=origin[i];
			box[1][0]=box[2][0]=box[2][1]=0.0;
			ptr = fgets(line,MAXLINE,fid);//ITEM: ATOMS id type x y z c_energy fx fy fz sx sy sz
			nwords = 0;
			char *words[count_words(line)+1];
			words[nwords++] = strtok(line," ,\t\n");
			while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			int colid = -1;
			int columnmap[10];
			for (int i=0;i<nwords-2;i++){columnmap[i]=-1;}
			for (int i=2;i<nwords;i++){
				if (strcmp(words[i],"type")==0){colid = 0;}
				else if (strcmp(words[i],"x")==0){colid=1;}
				else if (strcmp(words[i],"y")==0){colid=2;}
				else if (strcmp(words[i],"z")==0){colid=3;}
				//else if (strcmp(words[i],"c_energy")==0){colid=4;}
				else if (strcmp(words[i],"fx")==0){colid=4;}
				else if (strcmp(words[i],"fy")==0){colid=5;}
				else if (strcmp(words[i],"fz")==0){colid=6;}
				else if (strcmp(words[i],"sx")==0){colid=7;}
				else if (strcmp(words[i],"sy")==0){colid=8;}
				else if (strcmp(words[i],"sz")==0){colid=9;}
				else {continue;}
				cols[colid] = true;
				if (colid!=-1){columnmap[colid]=i-2;}
			}
			for (int i=0;i<4;i++){
				if (!cols[i]){errorf("dump file must include type, x, y, and z data columns (other recognized keywords are fx, fy, fz, sx, sy, sz)");}
			}
			bool doforce = false;
			bool dospin = false;
			sims[file].inum = natoms;
			sims[file].ilist = new int [natoms];
			sims[file].type = new int [natoms];
			sims[file].x= new double *[natoms];
			for (int i=0;i<3;i++){
				for (int j=0;j<3;j++)sims[file].box[i][j]=box[i][j];
				sims[file].origin[i]=origin[i];
			}
			//sims[file].energy = new double [natoms];
			for (int i=0;i<natoms;i++){
				sims[file].x[i]=new double [3];
			}
			//if force calibration is on
			if (doforces){
				sims[file].f = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].f[i] = new double [3];
				}
			}
			//if forces are given in dump file
			if (cols[4] && cols[5] && cols[6] && doforces){
				doforce = true;
				sims[file].forces=doforce;
			}
			if (cols[7] && cols[8] && cols[9]){
				dospin = true;
				sims[file].s = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].s[i] = new double [3];
				}
			}
			else if (this->dospin){
				errorf("spin vectors must be defined for all input simulations when magnetic fingerprints are used\n");
			}
			for (int i=0;i<natoms;i++){
				ptr = fgets(line,MAXLINE,fid);
				char *words2[count_words(line)+1];
				nwords1 = 0;
				words2[nwords1++] = strtok(line," ,\t");
				while ((words2[nwords1++] = strtok(NULL," ,\t"))) continue;
				nwords1--;
				if (nwords1!=nwords-2){errorf("incorrect number of data columns in dump file.");}
				sims[file].ilist[i]=i;//ignore any id mapping in the dump file, just id them based on line number.
				sims[file].type[i]=strtol(words2[columnmap[0]],NULL,10)-1;//lammps type counting starts at 1 instead of 0
				sims[file].x[i][0]=strtod(words2[columnmap[1]],NULL);
				sims[file].x[i][1]=strtod(words2[columnmap[2]],NULL);
				sims[file].x[i][2]=strtod(words2[columnmap[3]],NULL);
				//sims[file].energy[i]=strtod(words[columnmap[4]],NULL);
				if (doforce){
					sims[file].f[i][0]=strtod(words2[columnmap[4]],NULL);
					sims[file].f[i][1]=strtod(words2[columnmap[5]],NULL);
					sims[file].f[i][2]=strtod(words2[columnmap[6]],NULL);
				}
				//if force calibration is on, but forces are not given in file, assume they are zero.
				else if (doforces){
					sims[file].f[i][0]=0.0;
					sims[file].f[i][1]=0.0;
					sims[file].f[i][2]=0.0;
				}
				if (dospin){
					sims[file].s[i][0]=strtod(words2[columnmap[7]],NULL);
					sims[file].s[i][1]=strtod(words2[columnmap[8]],NULL);
					sims[file].s[i][2]=strtod(words2[columnmap[9]],NULL);
				}
				sims[file].spins = dospin;
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
			file++;
			if (file>nsims){errorf("Too many dump files found. Nsims is incorrect.\n");}
		}
		fclose(fid);
	}

	closedir(folder);
	sprintf(line,"imported %d atoms, %d simulations\n",natoms,nsims);
	std::cout<<line;
}

//part of setup. Do not optimize:
void PairRANN::create_neighbor_lists(){
	//brute force search technique rather than tree search because we only do it once and most simulations are small.
	//I did optimize for low memory footprint by only adding ghost neighbors
	//within cutoff distance of the box
	int i,ix,iy,iz,j,k;
//	char str[MAXLINE];
	double buffer = 0.01;//over-generous compensation for roundoff error
	std::cout<<"building neighbor lists\n";
	for (i=0;i<nsims;i++){
		double box[3][3];
		for (ix=0;ix<3;ix++){
			for (iy=0;iy<3;iy++)box[ix][iy]=sims[i].box[ix][iy];
		}
		double *origin = sims[i].origin;
		int natoms = sims[i].inum;
		int xb = floor(cutmax/box[0][0]+1);
		int yb = floor(cutmax/box[1][1]+1);
		int zb = floor(cutmax/box[2][2]+1);
		int buffsize = natoms*(xb*2+1)*(yb*2+1)*(zb*2+1);
		double x[buffsize][3];
		int type[buffsize];
		int id[buffsize];
		double spins[buffsize][3];
		int count = 0;

		//force all atoms to be inside the box:
		double xtemp[3];
		double xp[3];
		double boxt[9];
		for (j=0;j<3;j++){
			for (k=0;k<3;k++){
				boxt[j*3+k]=box[j][k];
			}
		}
		for (j=0;j<natoms;j++){
			for (k=0;k<3;k++){
				xp[k] = sims[i].x[j][k]-origin[k];
			}
			qrsolve(boxt,3,3,xp,xtemp);//convert coordinates from Cartesian to box basis (uses qrsolve for matrix inversion)
			for (k=0;k<3;k++){
				xtemp[k]-=floor(xtemp[k]);//if atom is outside box find periodic replica in box
			}
			for (k=0;k<3;k++){
				sims[i].x[j][k] = 0.0;
				for (int l=0;l<3;l++){
					sims[i].x[j][k]+=box[k][l]*xtemp[l];//convert back to Cartesian
				}
				sims[i].x[j][k]+=origin[k];
			}
		}

		//calculate box face normal directions and plane intersections
		double xpx,xpy,xpz,ypx,ypy,ypz,zpx,zpy,zpz;
		zpx = 0;zpy=0;zpz =1;
		double ym,xm;
		ym = sqrt(box[1][2]*box[1][2]+box[2][2]*box[2][2]);
		xm = sqrt(box[1][1]*box[2][2]*box[1][1]*box[2][2]+box[0][1]*box[0][1]*box[2][2]*box[2][2]+(box[0][1]*box[1][2]-box[0][2]*box[1][1])*(box[0][1]*box[1][2]-box[0][2]*box[1][1]));
		//unit vectors normal to box faces:
		ypx = 0;
		ypy = box[2][2]/ym;
		ypz = -box[1][2]/ym;
		xpx = box[1][1]*box[2][2]/xm;
		xpy = -box[0][1]*box[2][2]/xm;
		xpz = (box[0][1]*box[1][2]-box[0][2]*box[1][1])/xm;
		double fxn,fxp,fyn,fyp,fzn,fzp;
		//minimum distances from origin to planes aligned with box faces:
		fxn = origin[0]*xpx+origin[1]*xpy+origin[2]*xpz;
		fyn = origin[0]*ypx+origin[1]*ypy+origin[2]*ypz;
		fzn = origin[0]*zpx+origin[1]*zpy+origin[2]*zpz;
		fxp = (origin[0]+box[0][0])*xpx+(origin[1]+box[1][0])*xpy+(origin[2]+box[2][0])*xpz;
		fyp = (origin[0]+box[0][1])*ypx+(origin[1]+box[1][1])*ypy+(origin[2]+box[2][1])*ypz;
		fzp = (origin[0]+box[0][2])*zpx+(origin[1]+box[1][2])*zpy+(origin[2]+box[2][2])*zpz;
		//fill buffered atom list
		double px,py,pz;
		double xe,ye,ze;
		double theta,sx,sy,sz;
		for (j=0;j<natoms;j++){
			x[count][0] = sims[i].x[j][0];
			x[count][1] = sims[i].x[j][1];
			x[count][2] = sims[i].x[j][2];
			type[count] = sims[i].type[j];
			if (sims[i].spins){
				spins[count][0] = sims[i].s[j][0];
				spins[count][1] = sims[i].s[j][1];
				spins[count][2] = sims[i].s[j][2];
			}
			id[count] = j;
			count++;
		}

		//add ghost atoms outside periodic boundaries:
		for (ix=-xb;ix<=xb;ix++){
			for (iy=-yb;iy<=yb;iy++){
				for (iz=-zb;iz<=zb;iz++){
					if (ix==0 && iy == 0 && iz == 0)continue;
					for (j=0;j<natoms;j++){
						xe = ix*box[0][0]+iy*box[0][1]+iz*box[0][2]+sims[i].x[j][0];
						ye = iy*box[1][1]+iz*box[1][2]+sims[i].x[j][1];
						ze = iz*box[2][2]+sims[i].x[j][2];
						px = xe*xpx+ye*xpy+ze*xpz;
						py = xe*ypx+ye*ypy+ze*ypz;
						pz = xe*zpx+ye*zpy+ze*zpz;
						//include atoms if their distance from the box face is less than cutmax
						if (px>cutmax+fxp+buffer || px<fxn-cutmax-buffer){continue;}
						if (py>cutmax+fyp+buffer || py<fyn-cutmax-buffer){continue;}
						if (pz>cutmax+fzp+buffer || pz<fzn-cutmax-buffer){continue;}
						x[count][0] = xe;
						x[count][1] = ye;
						x[count][2] = ze;
						type[count] = sims[i].type[j];
						id[count] = j;
						if (sims[i].spinspirals && sims[i].spins){
							// sims[i].s[j][0]=1;
							// sims[i].s[j][2]=0;
							// spins[j][0]=1;
							// spins[j][2]=0;
							theta = sims[i].spinvec[0]*(ix*box[0][0]+iy*box[0][1]+iz*box[0][2]) + sims[i].spinvec[1]*(iy*box[1][1]+iz*box[1][2]) + sims[i].spinvec[2]*iz*box[2][2];
							double sxi = sims[i].s[j][0];
							double syi = sims[i].s[j][1];
							double szi = sims[i].s[j][2];
							double ax = sims[i].spinaxis[0];
							double ay = sims[i].spinaxis[1];
							double az = sims[i].spinaxis[2];
							ax = 0;//REMOVE
							az = 1;//REMOVE
							sx = sxi*(ax*ax*(1-cos(theta))+cos(theta))+syi*(ax*ay*(1-cos(theta))-az*sin(theta))+szi*(ax*az*(1-cos(theta))+ay*sin(theta));
							sy = sxi*(ax*ay*(1-cos(theta))+az*sin(theta))+syi*(ay*ay*(1-cos(theta))+cos(theta))+szi*(-ax*sin(theta)+ay*az*(1-cos(theta)));
							sz = sxi*(ax*az*(1-cos(theta))-ay*sin(theta))+syi*(ax*sin(theta)+ay*az*(1-cos(theta)))+szi*(az*az*(1-cos(theta))+cos(theta));
							//sx = ax*(ax*sxi+ay*syi+az*szi)+(ay*szi-az*syi)*sin(theta)+(-ay*(ax*syi-ay*sxi)+az*(-ax*szi+az*sxi))*cos(theta);
							//sy = ay*(ax*sxi+ay*syi+az*szi)+(-ax*szi+az*sxi)*sin(theta)+(ax*(ax*syi-ay*sxi)-az*(ay*szi-az*syi))*cos(theta);
							//sz = az*(ax*sxi+ay*syi+az*szi)+(ax*syi-ay*sxi)*sin(theta)+(-ax*(-ax*szi+az*sxi)-ay*(ay*szi-az*syi))*cos(theta);
							spins[count][0]=sx;
							spins[count][1]=sy;
							spins[count][2]=sz;
						}
						else if (sims[i].spins) {
							spins[count][0]=sims[i].s[j][0];
							spins[count][1]=sims[i].s[j][1];
							spins[count][2]=sims[i].s[j][2];
						}
						count++;
						if (count>buffsize){errorf("neighbor overflow!\n");}
					}
				}
			}
		}

		//update stored lists
		buffsize = count;
		for (j=0;j<natoms;j++){
			delete [] sims[i].x[j];
		}
		delete [] sims[i].x;
		delete [] sims[i].type;
		delete [] sims[i].ilist;
		if (sims[i].spins){
			for (j=0;j<natoms;j++){
				delete [] sims[i].s[j];
			}
			delete [] sims[i].s;
			sims[i].s = new double *[buffsize];
		}
		sims[i].type = new int [buffsize];
		sims[i].x = new double *[buffsize];
		sims[i].id = new int [buffsize];
		sims[i].ilist = new int [buffsize];

		for (j=0;j<buffsize;j++){
			sims[i].x[j] = new double [3];
			for (k=0;k<3;k++){
				sims[i].x[j][k] = x[j][k];
			}
			sims[i].type[j] = type[j];
			sims[i].id[j] = id[j];
			sims[i].ilist[j] = j;
			if (sims[i].spins){
				sims[i].s[j] = new double [3];
				for (k=0;k<3;k++){
					sims[i].s[j][k]=spins[j][k];
				}
			}
		}
		sims[i].inum = natoms;
		sims[i].gnum = buffsize-natoms;
		sims[i].numneigh = new int[natoms];
		sims[i].firstneigh = new int*[natoms];
		//do double count, slow, but enables getting the exact size of the neighbor list before filling it.
		for (j=0;j<natoms;j++){
			sims[i].numneigh[j]=0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].numneigh[j]++;
				}
			}
			sims[i].firstneigh[j] = new int[sims[i].numneigh[j]];
			count = 0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].firstneigh[j][count] = k;
					count++;
				}
			}
		}
	}
}

//part of setup. Do not optimize:
//TO DO: fix stack size problem
void PairRANN::compute_fingerprints(){
	std::cout<<"computing fingerprints\n";
	int nn,j,ii,f,i,itype,jnum;
	for (nn=0;nn<nsims;nn++){
		sims[nn].features = new double *[sims[nn].inum];
		sims[nn].state_e = 0;
		if (doforces){
			sims[nn].dfx = new double *[sims[nn].inum];
			sims[nn].dfy = new double *[sims[nn].inum];
			sims[nn].dfz = new double *[sims[nn].inum];
			if (dospin){
				sims[nn].dsx = new double *[sims[nn].inum];
				sims[nn].dsy = new double *[sims[nn].inum];
				sims[nn].dsz = new double *[sims[nn].inum];
			}
		}
		sims[nn].force = new double*[sims[nn].inum+sims[nn].gnum];
		sims[nn].fm = new double*[sims[nn].inum+sims[nn].gnum];
		  for (j=0;j<sims[nn].inum+sims[nn].gnum;j++){
			  sims[nn].force[j]=new double[3];
			  sims[nn].fm[j]=new double[3];
			  sims[nn].force[j][0]=0;
			  sims[nn].force[j][1]=0;
			  sims[nn].force[j][2]=0;
			  sims[nn].fm[j][0]=0;
			  sims[nn].fm[j][1]=0;
			  sims[nn].fm[j][2]=0;
		  }
			for (ii=0;ii<sims[nn].inum;ii++){
				i = sims[nn].ilist[ii];
			  	itype = map[sims[nn].type[i]];
			    f = net[itype].dimensions[0];
				jnum = sims[nn].numneigh[i];
				sims[nn].features[ii] = new double [f];
				if (doforces){
				  sims[nn].dfx[ii] = new double[f*jnum];
				  sims[nn].dfy[ii] = new double[f*jnum];
				  sims[nn].dfz[ii] = new double[f*jnum];
				  if (dospin){
					  sims[nn].dsx[ii] = new double[f*jnum];
					  sims[nn].dsy[ii] = new double[f*jnum];
					  sims[nn].dsz[ii] = new double[f*jnum];
				  }
			   }
		  }
		}
		#pragma omp parallel
		{
		int i,ii,itype,f,jnum,len,j,nn;
		double **force,**fm;
		#pragma omp for schedule(guided)
		for (nn=0;nn<nsims;nn++){
		  clock_t start = clock();
		
		  double start_time = omp_get_wtime();
		  force = sims[nn].force;
		  fm = sims[nn].fm;
		  if (debug_level2_freq>0){
			  sims[nn].state_ea = new double [sims[nn].inum];
		  }
		  for (ii=0;ii<sims[nn].inum;ii++){
			  i = sims[nn].ilist[ii];
			  itype = map[sims[nn].type[i]];
			  f = net[itype].dimensions[0];
			  jnum = sims[nn].numneigh[i];
			  double xn[jnum];
			  double yn[jnum];
			  double zn[jnum];
			  int tn[jnum];
			  int jl[jnum];
			  cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn);
			  double features [f];
			  double dfeaturesx[f*jnum];
			  double dfeaturesy[f*jnum];
			  double dfeaturesz[f*jnum];
			  for (j=0;j<f;j++){
				  features[j]=0;
			  }
			  for (j=0;j<f*jnum;j++){
				  dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
			  }
			  //screening is calculated once for all atoms if any fingerprint uses it.
			  double Sik[jnum];
			  double dSikx[jnum];
			  double dSiky[jnum];
			  double dSikz[jnum];
			  //TO D0: stack overflow often happens here from stack limit too low.
			  double dSijkx[jnum*jnum];
			  double dSijky[jnum*jnum];
			  double dSijkz[jnum*jnum];
			  //TO D0: stack overflow often happens here from stack limit too low.
			  bool Bij[jnum];
			  double sx[jnum*f];
			  double sy[jnum*f];
			  double sz[jnum*f];
			  for (j=0;j<f*jnum;j++){
				  sx[j]=sy[j]=sz[j]=0;
			  }
			  if (doscreen){
					screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);//jnum is neighlist + self term, hence jnum-1 in function inputs
			  }
			  if (allscreen){
				  screen_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,Bij,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz);
			  }
			  //do fingerprints for atom type
			  len = fingerprintperelement[itype];
			  for (j=0;j<len;j++){
				  	  	   if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			  }
			  itype = nelements;
			  //do fingerprints for type "all"
			  len = fingerprintperelement[itype];
			  for (j=0;j<len;j++){
				  	  	   if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				  	  else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			  }
			  //copy features from stack to heap
			  for (j=0;j<f;j++){
				  sims[nn].features[ii][j] = features[j];
			  }
			  if (doforces){
				  for (j=0;j<f*jnum;j++){
					  sims[nn].dfx[ii][j]=dfeaturesx[j];
					  sims[nn].dfy[ii][j]=dfeaturesy[j];
					  sims[nn].dfz[ii][j]=dfeaturesz[j];
				  }
				  if (dospin){
					  for (j=0;j<f*jnum;j++){
						  sims[nn].dsx[ii][j] = sx[j];
						  sims[nn].dsy[ii][j] = sy[j];
						  sims[nn].dsz[ii][j] = sz[j];
					  }
				  }
			  }
			  double e=0.0;
			  itype = map[sims[nn].type[i]];
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  itype = nelements;
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  sims[nn].energy-=e;
			  sims[nn].state_e+=e;
			  if (debug_level2_freq>0){sims[nn].state_ea[ii]=e;}
		  }
		  clock_t end = clock();
		  sims[nn].time = (double)(end-start)/ CLOCKS_PER_SEC;
	}
	}
}

void PairRANN::normalize_data(){
	int i,n,ii,j,itype;
	int natoms[nelementsp];
	normalgain = new double *[nelementsp];
	normalshift = new double *[nelementsp];
	//initialize
	for (i=0;i<nelementsp;i++){
		if (net[i].layers==0)continue;
		normalgain[i] = new double [net[i].dimensions[0]];
		normalshift[i] = new double [net[i].dimensions[0]];
		for (j=0;j<net[i].dimensions[0];j++){
			normalgain[i][j]=0;
			normalshift[i][j]=0;
		}
		natoms[i] = 0;
	}
	//get mean value of each 1st layer neuron input
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			natoms[itype]++;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalshift[itype][j]+=sims[n].features[ii][j];
				}
			}
			itype = nelements;
			natoms[itype]++;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalshift[itype][j]+=sims[n].features[ii][j];
				}
			}
		}
	}
	for (i=0;i<nelementsp;i++){
		if (net[i].layers==0)continue;
		for (j=0;j<net[i].dimensions[0];j++){
			normalshift[i][j]/=natoms[i];
		}
	}
	//get standard deviation
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalgain[itype][j]+=(sims[n].features[ii][j]-normalshift[itype][j])*(sims[n].features[ii][j]-normalshift[itype][j]);
				}
			}
			itype = nelements;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					normalshift[itype][j]+=(sims[n].features[ii][j]-normalshift[itype][j])*(sims[n].features[ii][j]-normalshift[itype][j]);
				}
			}
		}
	}
	for (i=0;i<nelementsp;i++){
		if (net[i].layers==0)continue;
		for (j=0;j<net[i].dimensions[0];j++){
			normalgain[i][j]=sqrt(normalgain[i][j]/natoms[i]);
		}
	}
	//shift input to mean=0, std = 1
	for (n=0;n<nsims;n++){
		for (ii=0;ii<sims[n].inum;ii++){
			itype = sims[n].type[ii];
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					if (normalgain[itype][j]>0){
						sims[n].features[ii][j] -= normalshift[itype][j];
						sims[n].features[ii][j] /= normalgain[itype][j];
					}
				}
			}
			itype = nelements;
			if (net[itype].layers!=0){
				for (j=0;j<net[itype].dimensions[0];j++){
					if (normalgain[itype][j]>0){
						sims[n].features[ii][j] -= normalshift[itype][j];
						sims[n].features[ii][j] /= normalgain[itype][j];
					}
				}
			}
		}
	}
	NNarchitecture *net_new = new NNarchitecture[nelementsp];
	normalize_net(net_new);
	copy_network(net_new,net);
	delete [] net_new;
}

void PairRANN::unnormalize_net(NNarchitecture *net_out){
	int i,j,k;
	double temp;
	copy_network(net,net_out);
	for (i=0;i<nelementsp;i++){
		if (net[i].layers>0){
			for (int i1=0;i1<net[i].bundles[0];i1++){
			for (j=0;j<net[i].bundleoutputsize[0][i1];j++){
				temp = 0.0;
				for (k=0;k<net[i].bundleinputsize[0][i1];k++){
					if (normalgain[i][k]>0){
						net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]/=normalgain[i][net[i].bundleinput[0][i1][k]];
						temp+=net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]*normalshift[i][net[i].bundleinput[0][i1][k]];
					}
				}
				net_out[i].bundleB[0][i1][j]-=temp;
			}
			}
		}
	}
}

void PairRANN::normalize_net(NNarchitecture *net_out){
	int i,j,k;
	double temp;
	copy_network(net,net_out);
	for (i=0;i<nelementsp;i++){
		if (net[i].layers>0){
			for (int i1=0;i1<net[i].bundles[0];i1++)
			for (j=0;j<net[i].bundleoutputsize[0][i1];j++){
				temp = 0.0;
				for (k=0;k<net[i].bundleinputsize[0][i1];k++){
					if (normalgain[i][k]>0){
						temp+=net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]*normalshift[i][net[i].bundleinput[0][i1][k]];
						if (weightdefined[i][i1][0])net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]*=normalgain[i][net[i].bundleinput[0][i1][k]];
					}
				}
				if (biasdefined[i][i1][0])net_out[i].bundleB[0][i1][j]+=temp;
			}
		}
	}
}

void PairRANN::separate_validation(){
	int n1,n2,i,vnum,len,startI,endI,j,t,k;
	char str[MAXLINE];
	int Iv[nsims];
	int Ir[nsims];
	bool w;
	n1=n2=0;
	sprintf(str,"finishing setup\n");
	std::cout<<str;
	for (i=0;i<nsims;i++)Iv[i]=-1;
	for (i=0;i<nsets;i++){
		startI=0;
		for (j=0;j<i;j++)startI+=Xset[j];
		endI = startI+Xset[i];
		len = Xset[i];
		// vnum = rand();
		// if (vnum<floor(RAND_MAX*validation)){
		// 	vnum = 1;
		// }
		// else{
		// 	vnum = 0;
		// }
		vnum = 0;// if Xset has only 1 entry, do not include it in validation ever. (Code above puts it randomly in validation or fit).
		vnum+=floor(len*validation);
		while (vnum>0){
			w = true;
			t = floor(rand() % len)+startI;
			for (j=0;j<n1;j++){
				if (t==Iv[j]){
					w = false;
					break;
				}
			}
			if (w){
				Iv[n1]=t;
				vnum--;
				n1++;
			}
		}
		for (j=startI;j<endI;j++){
			w = true;
			for (k=0;k<n1;k++){
				if (j==Iv[k]){
					w = false;
					break;
				}
			}
			if (w){
				Ir[n2]=j;
				n2++;
			}
		}
	}
	nsimr = n2;
	nsimv = n1;
	r = new int [n2];
	v = new int [n1];
	natomsr = 0;
	natomsv = 0;
	for (i=0;i<n1;i++){
		v[i]=Iv[i];
		natomsv += sims[v[i]].inum;
	}
	for (i=0;i<n2;i++){
		r[i]=Ir[i];
		natomsr += sims[r[i]].inum;
	}
	sprintf(str,"assigning %d simulations (%d atoms) for validation, %d simulations (%d atoms) for fitting\n",nsimv,natomsv,nsimr,natomsr);
	std::cout<<str;
}

void PairRANN::copy_network(NNarchitecture *net_old,NNarchitecture *net_new){
	int i,j,k;
	for (i=0;i<nelementsp;i++){
		net_new[i].layers = net_old[i].layers;
		if (net_new[i].layers>0){
			net_new[i].maxlayer = net_old[i].maxlayer;
			net_new[i].sumlayers=net_old[i].sumlayers;
			net_new[i].dimensions = new int [net_new[i].layers];
			net_new[i].startI = new int [net_new[i].layers];
			net_new[i].bundleW = new double**[net_new[i].layers-1];
			net_new[i].bundleB = new double**[net_new[i].layers-1];
			net_new[i].freezeW = new bool**[net_new[i].layers-1];
			net_new[i].freezeB = new bool**[net_new[i].layers-1];
			net_new[i].bundleinputsize = new int*[net_new[i].layers-1];
			net_new[i].bundleoutputsize = new int*[net_new[i].layers-1];
			net_new[i].bundleinput = new int**[net_new[i].layers-1];
			net_new[i].bundleoutput = new int**[net_new[i].layers-1];
			net_new[i].bundles = new int [net_new[i].layers-1];
			net_new[i].identitybundle = new bool *[net_new[i].layers-1];
			for (j=0;j<net_old[i].layers;j++){
				net_new[i].dimensions[j]=net_old[i].dimensions[j];
				net_new[i].startI[j]=net_old[i].startI[j];
				if (j==net_old[i].layers-1)continue;
				net_new[i].bundles[j]=net_old[i].bundles[j];
				net_new[i].bundleW[j] = new double*[net_new[i].bundles[j]];
				net_new[i].bundleB[j] = new double*[net_new[i].bundles[j]];
				net_new[i].freezeW[j] = new bool*[net_new[i].bundles[j]];
				net_new[i].freezeB[j] = new bool*[net_new[i].bundles[j]];
				net_new[i].identitybundle[j] = new bool[net_new[i].bundles[j]];
				net_new[i].bundleinputsize[j] = new int[net_new[i].bundles[j]];
				net_new[i].bundleoutputsize[j] = new int [net_new[i].bundles[j]];
				net_new[i].bundleinput[j] = new int*[net_new[i].bundles[j]];
				net_new[i].bundleoutput[j] = new int*[net_new[i].bundles[j]];
				for (int i1=0;i1<net_old[i].bundles[j];i1++){
					net_new[i].identitybundle[j][i1]=net_old[i].identitybundle[j][i1];
					net_new[i].bundleinputsize[j][i1]=net_old[i].bundleinputsize[j][i1];
					net_new[i].bundleoutputsize[j][i1]=net_old[i].bundleoutputsize[j][i1];
					net_new[i].bundleinput[j][i1] = new int[net_new[i].bundleinputsize[j][i1]];
					net_new[i].bundleoutput[j][i1] = new int[net_new[i].bundleoutputsize[j][i1]];
					net_new[i].bundleW[j][i1] = new double[net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1]];
					net_new[i].bundleB[j][i1] = new double[net_new[i].bundleoutputsize[j][i1]];
					net_new[i].freezeW[j][i1] = new bool[net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1]];
					net_new[i].freezeB[j][i1] = new bool[net_new[i].bundleoutputsize[j][i1]];
					for (int k=0;k<net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1];k++){
						net_new[i].bundleW[j][i1][k] = net_old[i].bundleW[j][i1][k];
						net_new[i].freezeW[j][i1][k] = net_old[i].freezeW[j][i1][k];
					}
					for (int k=0;k<net_new[i].bundleinputsize[j][i1];k++){
						net_new[i].bundleinput[j][i1][k]=net_old[i].bundleinput[j][i1][k];
					}
					for (int k=0;k<net_new[i].bundleoutputsize[j][i1];k++){
						net_new[i].bundleoutput[j][i1][k]=net_old[i].bundleoutput[j][i1][k];
						net_new[i].bundleB[j][i1][k]=net_old[i].bundleB[j][i1][k];
						net_new[i].freezeB[j][i1][k]=net_old[i].freezeB[j][i1][k];
					}
				}
			}
		}
	}
}



//top level run function, calls compute_jacobian and qrsolve. Cannot be parallelized.
void PairRANN::levenburg_marquardt_ch(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,jlen,i,jlenv,j,jlen2;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1;
	double lambda = lambda_initial;
	double vraise = lambda_increase;
	double vreduce = lambda_reduce;
	char line[MAXLINE];
    this->energy_fitv_best = 10^300;
	int i_off, j_off, j_offPi;
	double time1, time2;

	jlen = nsimr;
	jlenv = nsimv;
	if (doforces){
		jlen += natoms*3;
		jlenv += natoms*3;
	}
	jlen2 = jlen;
	if (doregularizer)jlen += betalen;//do not regulate last bias
	jlen1 = jlen;
		sprintf(str,"types=%d; betalen=%d; jlen1=%d; jlen2=%d, regularization:%d\n",nelementsp,betalen,jlen1,jlen2, doregularizer);
	std::cout<<str;
	double J[jlen1*betalen];
	double J1[jlen1*betalen];
	double J2[betalen*betalen];
	double t2[betalen];
	double target[jlen1];
	double target1[jlen1];
	double targetv[jlenv];
	double beta[betalen];
	double beta1[betalen];
	double D[betalen];
	double *dp;
	double delta[jlen1];//extra length used internally in qrsolve
	dp = delta;
//	double *Jp = J;
//	double *Jp1 = J1;
    double *tp,*tp1,*bp,*bp1,*Jp,*Jp1;
	tp = target;
	tp1 = target1;
	bp = beta;
	bp1 = beta1;
	Jp = J;
	Jp1 = J1;
	force_fit = energy_fit = reg_fit = energy_fit1 = force_fit1 = reg_fit1 = 0.0;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	jacobian_convolution(Jp,tp,r,nsimr,natomsr,net);

	NNarchitecture net1[nelementsp];
	copy_network(net,net1);
	for (i=0;i<nsimr;i++){
		energy_fit += tp[i]*tp[i];
	}
	energy_fit/=nsimr;
	if (doforces){
		for (i=nsimr;i<nsimr+natoms*3;i++)force_fit +=tp[i]*tp[i];
		force_fit/=natomsr*3;
	}
	if (doregularizer){
		for (i=1;i<betalen;i++){
			i_off = i+jlen-betalen;
			reg_fit +=tp[i_off]*tp[i_off];
		}
		reg_fit /= betalen;
	}
	double initial_reg = regularizer;
	double initial_eng = energy_fit;
	flatten_beta(net,bp);
	force_fitv = energy_fitv = 0.0;
	int counter = 0;
	int count = 0;
	int count1 = 0;
	int count2 = 0;
	int count3 = 0;
	int count4 = 0;
	int count5 = 0;
	int count6 = 0;
	iter = 0;
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces){
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	}
	else{
		sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
	}
	write_potential_file(true,line,0,initial_reg);
	double start2;
	while (iter<max_epochs){
		if (goodstep){
			if (nsimv>0){
				//do validation forward pass
				forward_pass(targetv,v,nsimv,net);
				if (debug_level1_freq>0)write_debug_level1(tp,targetv);
				//compute_jacobian(J1,targetv,v,nsimv,natomsv,net1);
				energy_fitv=0.0;
				for (i=0;i<nsimv;i++){
					energy_fitv += targetv[i]*targetv[i];
				}
				energy_fitv /= nsimv;
				if (doforces){
					force_fitv = 0.0;
					for (i=nsimv;i<natomsv*3+nsimv;i++){
						force_fitv += targetv[i]*targetv[i];
					}
					force_fitv/=(natomsv*3);
				}
			}
			else{
				energy_fitv = 0.0;
				force_fitv = 0.0;
			}

			// clock_t start2 = clock();
			start2 = omp_get_wtime();

			for (i=0;i<betalen;i++){
				i_off = i*betalen;
				for (int k=0;k<=i;k++){
					J2[i_off+k] = 0.0;
				}
			}


			// #pragma omp parallel default(none) shared(J2,Jp,jlen2,betalen, doregularizer)
			#pragma omp parallel
			{
			// loop reordered to remove the dependancy. Single thread calculation would be slow. Observed gain when thread more than around 8
			#pragma omp for
			for (int i=0;i<betalen;i++){
				int i_off = i*betalen;
				for (int k=0;k<=i;k++){
					for (int j=0;j<jlen2;j++){
						int j_off = j*betalen;
						int j_offPi = j_off+i;
						J2[i_off+k] += Jp[j_offPi]*Jp[j_off+k];
					}
				}
			}

			if (doregularizer){
				#pragma omp for
				for (int i=0;i<betalen;i++){
					int	i_off = i*betalen;
					int ij_off = jlen2*betalen + i_off;
					J2[i_off+i]+=Jp[ij_off+i]*Jp[ij_off+i];
				}
			}

			#pragma omp barrier
			#pragma omp single
			{
			for (int i=0;i<betalen;i++){
				D[i] = J2[i*betalen+i];
				if (D[i]==0){errorf(FLERR,"Jacobian is rank deficient!\n");}//one or more weight/bias has no effect on the computed energy of any of the atoms.
				if (doregularizer) // t2 can be initialized with 0 or derivative w.r.t. weight
					t2[i]=Jp[jlen2*betalen+i*betalen+i]*tp[jlen2+i];
				else
				    t2[i]=0;
			}
			}

			// loop splitting for threading. Initialization for t2 is done above.
			#pragma omp for
			for (int i=0;i<betalen;i++){
				// t2[i]=0;
				for (j=0;j<jlen2;j++){
					t2[i]+=Jp[j*betalen+i]*tp[j];
				}
			}

			}
			double adexp = 0.1;
			if (adaptive_regularizer) {regularizer *= sqrt(initial_reg/reg_fit*energy_fit);}
			if (regularizer<1e-8){
				doregularizer=false;
				jlen -=betalen;
				jlen1 = jlen;
			}
			// if (doregularizer){
			// 	for (int i=0;i<betalen;i++){
			// 		t2[i]+=Jp[jlen2*betalen+i*betalen+i]*tp[jlen2+i];
			// 	}
			// }

			// clock_t end = clock();
			// time2 = (double) (end-start2) / CLOCKS_PER_SEC * 1000.0;
			// sprintf(str,"loop: %f ms\n",time2);
			// std::cout<<str;
			double time = (double) (omp_get_wtime() - start2)*1000.0;
			// printf("loop: %f ms\n",time);
            //printf(" - best fit : %f \n\n\n",energy_fitv);

		}
        bool is_write_potential = false;
        if (count == potential_output_freq){
			count = 0;
			if ((energy_fitv*natomsv+energy_fit*natomsr)/natoms < this->energy_fitv_best) {
				this->energy_fitv_best = (energy_fitv*natomsv+energy_fit*natomsr)/natoms;
				is_write_potential = true;
			}
			else {
				count--;
			}
        }
		if (doforces){
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		}
		else{
			sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);
		}
		std::cout<<line;
		fprintf(fid,"%s",line);
		count++;
		count1++;
		count2++;
		count3++;
		count4++;
		count5++;
		count6++;
        if (is_write_potential){
            write_potential_file(true,line,iter,initial_reg);
        }
        if (count1==debug_level1_freq){
			write_debug_level1(tp,targetv);
			count1=0;
		}
		if (count2==debug_level2_freq){
			write_debug_level2(tp,targetv);
			count2=0;
		}
		if (count3==debug_level3_freq){
			//always print the ones computed last, regardless of whether the step was good.
			if (goodstep){
				write_debug_level3(Jp,tp,bp,dp);
			}
			else {
				write_debug_level3(Jp1,tp1,bp1,dp);
			}
			count3=0;
		}
		if (count4==debug_level4_freq){
			write_debug_level4(tp,targetv);
			count4=0;
		}
		// if (count5==debug_level5_freq){
		// 	write_debug_level5(tp,targetv);
		// 	count5=0;
		// }
		counter++;
		// if (count1 == potential_output_freq){
         //   if (energy_fitv < this->energy_fitv_best){
         //       this->energy_fitv_best = energy_fitv;
         //       printf(" so far best fit : %f \n\n\n",energy_fitv_best);
         //       write_potential_file(true,line);
         //   }
		//	count1 = 0;
		//}
		for (i=0;i<betalen;i++){
			J2[i*betalen+i]=D[i]+sqrt(D[i]*lambda);
		}

//		clock_t start1 = clock();
		chsolve(J2,betalen,t2,dp);
		// FILE *fidt = fopen("t2dpnew.log","w");
		// for (i=0;i<betalen;i++){
		// 	fprintf(fidt,"%f,%f\n",t2[i],dp[i]);
		// }
		// fclose(fidt);
		// fidt = fopen("J2new.log","w");
		// for (i=0;i<betalen;i++){
		// 	for (j=0;j<betalen;j++){
		// 			fprintf(fidt,"%f,",J2[i*betalen+j]);
		// 		}
		// 		fprintf(fidt,"\n");
		// }
		// fclose(fidt);
		// if (counter==1){errorf("stop");}
//		clock_t end1 = clock();
//		time = (double) (end1-start1) / CLOCKS_PER_SEC * 1000.0;
//		sprintf(str,"chsolve(): %f ms\n",time);
//		std::cout<<str;

		// for (i=0;i<betalen;i++){
		// 	sprintf(str,"i %d  %f %f\n",i,bp[i],dp[i]);
		// 	std::cout<<str;
		// }
		for (i=0;i<betalen;i++)bp1[i]=bp[i]+dp[i];
		unflatten_beta(net1,bp1);
		jacobian_convolution(Jp1,tp1,r,nsimr,natomsr,net1);
		energy_fit1 = 0.0;
		for (i=0;i<nsimr;i++)energy_fit1 += tp1[i]*tp1[i];
		//for (i=0;i<nsimr;i++)printf("%d %f\n",i,tp1[i]);
		energy_fit1/=nsimr;
		if (doforces){
 			force_fit1 = 0.0;
			for (i=nsimr;i<natomsr*3+nsimr;i++){
				force_fit1 += tp1[i]*tp1[i];
			}
			force_fit1/=natomsr*3;
		}
		if (doregularizer){
			reg_fit1 = 0.0;
			for (i=1;i<betalen;i++){
				i_off = i+jlen-betalen;
				reg_fit1 += tp1[i_off]*tp1[i_off];
			}
			reg_fit1 /= betalen;
		}
		if (energy_fit1+force_fit1+reg_fit1<energy_fit+force_fit+reg_fit){
			goodstep = true;
			lambda = lambda*vreduce;
			energy_fit = energy_fit1;
			force_fit = force_fit1;
			reg_fit = reg_fit1;
			double *tempb;
			tempb = bp;
			bp = bp1;
			bp1= tempb;
			double *tempJ;
			tempJ = Jp;
			Jp = Jp1;
			Jp1 = tempJ;
			double *tempT = tp;
			tp = tp1;
			tp1 = tempT;
			unflatten_beta(net,bp);
			iter++;
		}
		else {
			goodstep=false;
			lambda = lambda*vraise;
			if (lambda > 10e50){
				//write_potential_file(true,line,iter,initial_reg);
				errorf("Terminating because convergence is not making progress.\n");
			}
		}
		if (energy_fit+force_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			write_potential_file(true,line,iter,initial_reg);
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<=nelements;i++){
		if (net1[i].layers>0){
			for (int j=0;j<net1[i].layers-1;j++){
				delete [] net1[i].bundleinputsize[j];
				delete [] net1[i].bundleoutputsize[j];
				for (int k=0;k<net1[i].bundles[j];k++){
					delete [] net1[i].bundleinput[j][k];
					delete [] net1[i].bundleoutput[j][k];
					delete [] net1[i].bundleW[j][k];
					delete [] net1[i].bundleB[j][k];
					delete [] net1[i].freezeW[j][k];
					delete [] net1[i].freezeB[j][k];
				}
				delete [] net1[i].bundleinput[j];
				delete [] net1[i].bundleoutput[j];
				delete [] net1[i].bundleW[j];
				delete [] net1[i].bundleB[j];
				delete [] net1[i].freezeW[j];
				delete [] net1[i].freezeB[j];
			}
			delete [] net1[i].bundleinput;
			delete [] net1[i].bundleoutput;
			delete [] net1[i].bundleW;
			delete [] net1[i].bundleB;
			delete [] net1[i].freezeW;
			delete [] net1[i].freezeB;
			delete [] net1[i].dimensions;
			delete [] net1[i].startI;
		}
	}

	// clock_t end = clock();
    // time1 = (double) (end-start1) / CLOCKS_PER_SEC * 1000.0;
	// sprintf(str,"LM_ch(): %f ms\n",time1);
	// std::cout<<str;
    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
    // printf("LM_ch(): %f ms\n",time);

}


void PairRANN::flatten_beta(NNarchitecture *net,double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
	for (itype=0;itype<nelementsp;itype++){
		for (i=0;i<net[itype].layers-1;i++){
			for (int i1=0;i1<net[itype].bundles[i];i1++){
				if (net[itype].identitybundle[i][i1])continue;
				for (k1=0;k1<net[itype].bundleoutputsize[i][i1];k1++){
					for (k2=0;k2<net[itype].bundleinputsize[i][i1];k2++){
						if (net[itype].freezeW[i][i1][k1*net[itype].bundleinputsize[i][i1]+k2])continue;
						beta[count2]=net[itype].bundleW[i][i1][k1*net[itype].bundleinputsize[i][i1]+k2];
						count2++;
					}
					if (net[itype].freezeB[i][i1][k1])continue;
					beta[count2]=net[itype].bundleB[i][i1][k1];
					count2++;
				}
			}
		}
	}
}

void PairRANN::unflatten_beta(NNarchitecture *net,double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
	for (itype=0;itype<nelementsp;itype++){
		for (i=0;i<net[itype].layers-1;i++){
			for (int i1=0;i1<net[itype].bundles[i];i1++){
				if (net[itype].identitybundle[i][i1])continue;
				for (k1=0;k1<net[itype].bundleoutputsize[i][i1];k1++){
					for (k2=0;k2<net[itype].bundleinputsize[i][i1];k2++){
						if (net[itype].freezeW[i][i1][k1*net[itype].bundleinputsize[i][i1]+k2])continue;
						net[itype].bundleW[i][i1][k1*net[itype].bundleinputsize[i][i1]+k2]=beta[count2];
						count2++;
					}
					if (net[itype].freezeB[i][i1][k1])continue;
					net[itype].bundleB[i][i1][k1]=beta[count2];
					count2++;
				}
			}
		}
	}
}



void PairRANN::jacobian_convolution(double *J,double *target,int *s,int sn,int natoms,NNarchitecture *net){

	//clock_t start = clock();
	double start_time = omp_get_wtime();
	#pragma omp parallel
	{
//	char str[MAXLINE];
	int nn,ii,n1;
	int count4 = 0;
	int n1dimi, n1sl, n1slM1, n4s, lM1, pIPk2, pLPk2;
	int sPcPiiX3, p1dlxyz, p2dlxyz, jPstartI, jjXfPk, iiX3, j1X3, p1dXw, p2dXw, p1ddXw, p2ddXw, i2n1W;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = sims[nn].inum;
		double energy;
		double force[n4s*3];
		energy = 0.0;
		for (ii=0;ii<betalen;ii++){
			J[n1*betalen+ii]=0.0;
		}
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,startI,prevI,l,startL,prevL,k1,k2,k3;
			startI=0;
			NNarchitecture net1;
			itype = sims[nn].type[ii];
			net1 = net[itype];
			n1sl = net1.sumlayers;
			n1slM1 = n1sl-1;
			iiX3 = ii*3;
			sPcPiiX3 = sn+count4+iiX3;
			numneigh = sims[nn].numneigh[ii];
			jnum = numneigh+1;//extra value on the end of the array is the self term.
			firstneigh = sims[nn].firstneigh;
			jlist = firstneigh[ii];
			int L = net1.layers-1;
			double layer[n1sl];
			double dlayer[n1sl];
			double dlayerx[jnum*n1sl];
			double dlayery[jnum*n1sl];
			double dlayerz[jnum*n1sl];
			int f = net1.dimensions[0];
			double *features = sims[nn].features[ii];
			prevI = 0;
			n1dimi = net1.dimensions[0];
			for (k=0;k<n1dimi;k++){
				layer[k]=features[k];
				dlayer[k] = 1.0;
			}
			for (k=net1.dimensions[0];k<n1sl;k++){layer[k]=0;}
			for (i=0;i<net1.layers-1;i++){
				for (int i1 = 0;i1<net1.bundles[i];i1++){
					int s1 = net1.bundleoutputsize[i][i1];
					int s2 = net1.bundleinputsize[i][i1];
					for (j=0;j<s1;j++){
						startI = net1.startI[i+1];
						int j1 = net1.bundleoutput[i][i1][j];
						jPstartI = j1+startI;
						for (k=0;k<s2;k++){
							int k1 = net1.bundleinput[i][i1][k];
							layer[jPstartI] += net1.bundleW[i][i1][j*s2+k]*layer[k1+prevI];
						} 
						layer[jPstartI] += net1.bundleB[i][i1][j];
					}
				}
				for (j=0;j<net1.dimensions[i+1];j++){
					startI = net1.startI[i+1];
					jPstartI = j+startI;
					dlayer[jPstartI] = activation[itype][i][j]->dactivation_function(layer[jPstartI]);
					layer[jPstartI] =  activation[itype][i][j]-> activation_function(layer[jPstartI]);
					if (i==L-1){
						energy += layer[jPstartI];
					}
				}
				prevI = startI;
			}
			prevI=0;
			int count2=0;//skip frozen parameters
			int count3=0;//include frozen parameters
			if (itype>0){
				count2=betalen_v[itype-1];
				count3=betalen_f[itype-1];
			}
			//backpropagation
			for (i=0;i<net1.layers-1;i++){
				int d1 = net1.dimensions[i+1];
				int d2 = net1.dimensions[i];
				double dXw[d1*net1.sumlayers];
				startI = net1.startI[i+1];
				prevI = net1.startI[i];
				for (k1=0;k1<net1.sumlayers;k1++){
					for (k2=0;k2<d1;k2++){
						dXw[k1*d1+k2]=0.0;
						if (k1==k2+startI){
							dXw[k1*d1+k2]=1.0;
						}
					}
				}
				for (int i1=0;i1<net1.bundles[i];i1++){
					if (net1.identitybundle[i][i1])continue;
					int s3 = net1.bundleoutputsize[i][i1];
					int s4 = net1.bundleinputsize[i][i1];
					for (l=i+1;l<net1.layers-1;l++){
						int d3 = net1.dimensions[l+1];
						int d4 = net1.dimensions[l];
						int startL = net1.startI[l+1];
						int prevL = net1.startI[l];
						for (int l1=0;l1<net1.bundles[l];l1++){
							int s1 = net1.bundleoutputsize[l][l1];
							int s2 = net1.bundleinputsize[l][l1];
							for (k1=0;k1<s1;k1++){
								for (k2=0;k2<s2;k2++){
									for (k3=0;k3<s3;k3++){
										int p1 = net1.bundleoutput[l][l1][k1];
										int p2 = net1.bundleinput[l][l1][k2];
										int p3 = net1.bundleoutput[i][i1][k3];
										//dlayer_l/dlayer_i
										dXw[(p1+startL)*d1+p3]+=dlayer[p2+prevL]*net1.bundleW[l][l1][k1*s2+k2]*dXw[(p2+prevL)*d1+p3];
									}
								}
							}
						}
					}
					for (k1=0;k1<s3;k1++){
						int p1 = net1.bundleoutput[i][i1][k1];
						//weights
						for (k2=0;k2<s4;k2++){
							int p2 = net1.bundleinput[i][i1][k2];
							if (~freezebeta[count3]){
								J[n1*betalen+count2] += -dXw[(net1.sumlayers-1)*d1+p1]*layer[p2+prevI]*sims[nn].energy_weight;
								count2++;
							}
							count3++;
						}
						//bias
						if (~freezebeta[count3]){
							J[n1*betalen+count2] += -dXw[(net1.sumlayers-1)*d1+p1]*sims[nn].energy_weight;
							count2++;
						}
						count3++;
					}
				}
			}
		}
		//fill error vector
		target[n1] = (energy-sims[nn].energy)*sims[nn].energy_weight;
	}
	//regularizer
	if (doregularizer){
		int count2 = 0;
		int count3 = 0;
		// #pragma omp for schedule(dynamic)
		for (int itype=0;itype<nelementsp;itype++){
			for (int i=0;i<net[itype].layers-1;i++){
				for (int i1=0;i1<net[itype].bundles[i];i1++){
					if (net[itype].identitybundle[i][i1])continue;
					for (int k1=0;k1<net[itype].bundleoutputsize[i][i1];k1++){
						for (int k2=0;k2<net[itype].bundleinputsize[i][i1];k2++){
							if (~freezebeta[count3]){
								J[(sn+count2)*betalen+count2] = regularizer;
								target[sn+count2] = -regularizer*net[itype].bundleW[i][i1][k1*net[itype].bundleinputsize[i][i1]+k2];
								count2++;
							}
							count3++;
						}
						if (~freezebeta[count3]){
							J[(sn+count2)*betalen+count2] = regularizer;
							target[sn+count2] = -regularizer*net[itype].bundleB[i][i1][k1];
							//force last bias to not count toward regularization
							if (i+2==net[itype].layers){
								J[(sn+count2)*betalen+count2]=0;
								target[(sn+count2)*betalen+count2]=0;
							}
							count2++;
						}
						count3++;
					}
				}
			}
		}
	}

    }

//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	// printf(" - compute_jacobian(): %f ms\n",time);

}

//finds total error from features
void PairRANN::forward_pass(double *target,int *s,int sn,NNarchitecture *net){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	#pragma omp parallel
	{
	int nn,ii,n1;
	int jPstartI, jjXfPk, n1sl, n1slM1, p1dlxyz, p2dlxyz, sPcPiiX3, n4s, iiX3, j1X3;
	int count4 = 0;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = sims[nn].inum;
		double energy;
		energy = 0.0;
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,startI,prevI;
			startI=0;
			NNarchitecture net1;
			itype = sims[nn].type[ii];
			net1 = net[itype];
			n1sl = net1.sumlayers;
			n1slM1 = n1sl-1;
			numneigh = sims[nn].numneigh[ii];
			jnum = numneigh+1;//extra value on the end of the array is the self term.
			firstneigh = sims[nn].firstneigh;
			jlist = firstneigh[ii];
			int L = net1.layers-1;
			double layer[n1sl];
			double dlayer[n1sl];
			double dlayerx[jnum*n1sl];
			double dlayery[jnum*n1sl];
			double dlayerz[jnum*n1sl];
			int f = net1.dimensions[0];
			double *features = sims[nn].features[ii];
			double *dfeaturesx;
			double *dfeaturesy;
			double *dfeaturesz;
			if (doforces){
				dfeaturesx = sims[nn].dfx[ii];
				dfeaturesy = sims[nn].dfy[ii];
				dfeaturesz = sims[nn].dfz[ii];
			}
			prevI = 0;
			for (k=0;k<net1.dimensions[0];k++){
				layer[k]=features[k];
				dlayer[k] = 1.0;
			}
			for (k=net1.dimensions[0];k<n1sl;k++){layer[k]=0;}
			for (i=0;i<net1.layers-1;i++){
				for (int i1=0;i1<net1.bundles[i];i1++){
					int s1 = net1.bundleoutputsize[i][i1];
					int s2 = net1.bundleinputsize[i][i1];
					for (j=0;j<s1;j++){
						startI = net1.startI[i+1];
						j1 = net1.bundleoutput[i][i1][j];
						jPstartI = j1+startI;
						for (k=0;k<s2;k++){
							int k1 = net1.bundleinput[i][i1][k];
							layer[jPstartI] += net1.bundleW[i][i1][j*s2+k]*layer[k1+prevI];
						}
						layer[jPstartI] += net1.bundleB[i][i1][j];
					}	
				}
				for (j=0;j<net1.dimensions[i+1];j++){
					startI = net1.startI[i+1];
					jPstartI = j+startI;
					dlayer[jPstartI] = activation[itype][i][j]->dactivation_function(layer[jPstartI]);
					layer[jPstartI] =  activation[itype][i][j]-> activation_function(layer[jPstartI]);
					if (i==L-1){
						energy += layer[jPstartI];
					}
				}
				prevI=startI;
			}
			prevI=0;
			target[n1] = (energy-sims[nn].energy)*sims[nn].energy_weight;
		}
	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

//finds per atom energies from features
void PairRANN::get_per_atom_energy(double **energies,int *s,int sn,NNarchitecture *net){
	double start_time = omp_get_wtime();
	#pragma omp parallel
	{
	int nn,ii,n1;
	int jPstartI, jjXfPk, n1sl, n1slM1, p1dlxyz, p2dlxyz, sPcPiiX3, n4s, iiX3, j1X3;
	int count4 = 0;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = sims[nn].inum;
		energies[n1] = new double[n4s];
		double energy;
		energy = 0.0;
		for (ii=0;ii<n4s;ii++){
			energies[n1][ii]=0;
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,startI,prevI;
			startI=0;
			NNarchitecture net1;
			itype = sims[nn].type[ii];
			net1 = net[itype];
			n1sl = net1.sumlayers;
			n1slM1 = n1sl-1;
			numneigh = sims[nn].numneigh[ii];
			jnum = numneigh+1;//extra value on the end of the array is the self term.
			firstneigh = sims[nn].firstneigh;
			jlist = firstneigh[ii];
			int L = net1.layers-1;
			double layer[n1sl];
			double dlayer[n1sl];
			double dlayerx[jnum*n1sl];
			double dlayery[jnum*n1sl];
			double dlayerz[jnum*n1sl];
			int f = net1.dimensions[0];
			double *features = sims[nn].features[ii];
			double *dfeaturesx;
			double *dfeaturesy;
			double *dfeaturesz;
			dfeaturesx = sims[nn].dfx[ii];
			dfeaturesy = sims[nn].dfy[ii];
			dfeaturesz = sims[nn].dfz[ii];
			prevI = 0;
			for (k=0;k<net1.dimensions[0];k++){
				layer[k]=features[k];
				dlayer[k] = 1.0;
			}
			for (k=net1.dimensions[0];k<n1sl;k++){layer[k]=0;}
			for (i=0;i<net1.layers-1;i++){
				for (int i1 = 0;i1<net1.bundles[i];i1++){
					int s1 = net1.bundleoutputsize[i][i1];
					int s2 = net1.bundleinputsize[i][i1];
					for (j=0;j<s1;j++){
						startI = net1.startI[i+1];
						int j1 = net1.bundleoutput[i][i1][j];
						jPstartI = j1+startI;
						for (k=0;k<s2;k++){
							int k1 = net1.bundleinput[i][i1][k];
							layer[jPstartI] += net1.bundleW[i][i1][j*s2+k]*layer[k1+prevI];
						} 
						layer[jPstartI] += net1.bundleB[i][i1][j];
					}
				}
				for (j=0;j<net1.dimensions[i+1];j++){
					startI = net1.startI[i+1];
					jPstartI = j+startI;
					dlayer[jPstartI] = activation[itype][i][j]->dactivation_function(layer[jPstartI]);
					layer[jPstartI] =  activation[itype][i][j]-> activation_function(layer[jPstartI]);
					if (i==L-1){
						energy += layer[jPstartI];
						energies[n1][ii]=layer[jPstartI];
					}
				}
				prevI = startI;
			}
		}
	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

//finds total energy and per atom forces from features
void PairRANN::propagateforward(double *energy,double **force,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,int *jl,int nn) {
  int i,j,k,jj,j1,i1;
  NNarchitecture net1 = net[itype];
  int L = net1.layers-1;
  //energy output with forces from analytical derivatives
  double dsum1[net1.maxlayer];
  int f = net1.dimensions[0];
  double sum[net1.maxlayer];
  double layer[net1.maxlayer];
  double dlayersumx[jnum][net1.maxlayer];
  double dlayersumy[jnum][net1.maxlayer];
  double dlayersumz[jnum][net1.maxlayer];
  double dlayerx[jnum][net1.maxlayer];
  double dlayery[jnum][net1.maxlayer];
  double dlayerz[jnum][net1.maxlayer];
  for (k=0;k<net1.dimensions[0];k++){
	  layer[k]=features[k];
	  for (jj=0;jj<jnum;jj++){
		  dlayerx[jj][k]=dfeaturesx[jj*f+k];
		  dlayery[jj][k]=dfeaturesy[jj*f+k];
		  dlayerz[jj][k]=dfeaturesz[jj*f+k];
	  }
  }
  for (i=0;i<net1.layers-1;i++) {
	for (j=0;j<net1.dimensions[i+1];j++){
		sum[j]=0;
	}
	for (i1=0;i1<net1.bundles[i];i1++){
		int s1=net1.bundleoutputsize[i][i1];
		int s2=net1.bundleinputsize[i][i1];
		for (j=0;j<s1;j++){
			int j1 = net1.bundleoutput[i][i1][j];
			for (k=0;k<s2;k++){
				int k1 = net1.bundleinput[i][i1][k];
				sum[j1] += net1.bundleW[i][i1][j*s2+k]*layer[k1];
			}
			sum[j1]+= net1.bundleB[i][i1][j];
		}
	}
    for (j=0;j<net1.dimensions[i+1];j++) {
      dsum1[j] = activation[itype][i][j]->dactivation_function(sum[j]);
      sum[j] = activation[itype][i][j]->activation_function(sum[j]);
      if (i==L-1) {
        energy[j] = sum[j];
      }
      //force propagation
      for (jj=0;jj<jnum;jj++) {
        dlayersumx[jj][j]=0;
        dlayersumy[jj][j]=0;
        dlayersumz[jj][j]=0;
	  }
	}
	for (i1=0;i1<net1.bundles[i];i1++){
		int s1 = net1.bundleoutputsize[i][i1];
		int s2 = net1.bundleinputsize[i][i1];
		for (j=0;j<s1;j++){
			int j1 = net1.bundleoutput[i][i1][j];
			for (jj=0;jj<jnum;jj++){
				for (k=0;k<s2;k++){
					int k1= net1.bundleinput[i][i1][k];
					double w1 = net1.bundleW[i][i1][j*s2+k];
					dlayersumx[jj][j1] += w1*dlayerx[jj][k1];
					dlayersumy[jj][j1] += w1*dlayery[jj][k1];
					dlayersumz[jj][j1] += w1*dlayerz[jj][k1];
				}
			}
		}
	}
	for (j=0;j<net1.dimensions[i+1];j++){
		for (jj=0;jj<jnum;jj++){
			dlayersumx[jj][j]*= dsum1[j];
			dlayersumy[jj][j]*= dsum1[j];
			dlayersumz[jj][j]*= dsum1[j];
		}
	}
	if (i==L-1) {
		for (j=0;j<net1.dimensions[i+1];j++){
			for (jj=0;jj<jnum-1;jj++){
				int j2 = jl[jj];
				force[j2][0]+=dlayersumx[jj][j];
				force[j2][1]+=dlayersumy[jj][j];
				force[j2][2]+=dlayersumz[jj][j];
			}
			int j2 = sims[nn].ilist[ii];
			jj = jnum-1;
			force[j2][0]+=dlayersumx[jj][j];
			force[j2][1]+=dlayersumy[jj][j];
			force[j2][2]+=dlayersumz[jj][j];
		}
	}
    //update values for next iteration
    for (j=0;j<net1.dimensions[i+1];j++) {
      layer[j]=sum[j];
      for (jj=0;jj<jnum;jj++) {
        dlayerx[jj][j] = dlayersumx[jj][j];
        dlayery[jj][j] = dlayersumy[jj][j];
        dlayerz[jj][j] = dlayersumz[jj][j];
      }
    }
  }
}

void PairRANN::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn){
	int *jlist,j,count,jj,*type,jtype;
	double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double **x = sims[sn].x;
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	type = sims[sn].type;
	jlist = sims[sn].firstneigh[i];
	count = 0;
	for (jj=0;jj<jnum[0];jj++){
		j = jlist[jj];
		j &= NEIGHMASK;
		jtype = map[type[j]];
		delx = xtmp - x[j][0];
		dely = ytmp - x[j][1];
		delz = ztmp - x[j][2];
		rsq = delx*delx + dely*dely + delz*delz;
		if (rsq>cutmax*cutmax){
			continue;
		}
		xn[count]=delx;
		yn[count]=dely;
		zn[count]=delz;
		tn[count]=jtype;
		//jl[count]=sims[sn].id[j];
		jl[count]=j;
		//jl is currently only used to calculate spin dot products.
		//j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
		//lammps code uses id instead of j because spin spirals are not supported.
		count++;
	}
	jnum[0]=count+1;
}

void PairRANN::screen_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,bool *Bij,double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz){
	double xnc[jnum[0]],ync[jnum[0]],znc[jnum[0]];
	double Sikc[jnum[0]];
	double dSikxc[jnum[0]];
	double dSikyc[jnum[0]];
	double dSikzc[jnum[0]];
	double dSijkxc[jnum[0]][jnum[0]];
	double dSijkyc[jnum[0]][jnum[0]];
	double dSijkzc[jnum[0]][jnum[0]];
	int jj,kk,count,count1,tnc[jnum[0]],jlc[jnum[0]];
	count = 0;
	for (jj=0;jj<jnum[0]-1;jj++){
		if (Bij[jj]){
			count1 = 0;
			xnc[count]=xn[jj];
			ync[count]=yn[jj];
			znc[count]=zn[jj];
			tnc[count]=tn[jj];
			jlc[count]=jl[jj];
			Sikc[count]=Sik[jj];
			dSikxc[count]=dSikx[jj];
			dSikyc[count]=dSiky[jj];
			dSikzc[count]=dSikz[jj];
			for (kk=0;kk<jnum[0]-1;kk++){
				if (Bij[kk]){
					dSijkxc[count][count1] = dSijkx[jj*(jnum[0]-1)+kk];
					dSijkyc[count][count1] = dSijky[jj*(jnum[0]-1)+kk];
					dSijkzc[count][count1] = dSijkz[jj*(jnum[0]-1)+kk];
					count1++;
				}
			}
			count++;
		}
	}
	jnum[0]=count+1;
	for (jj=0;jj<count;jj++){
		xn[jj]=xnc[jj];
		yn[jj]=ync[jj];
		zn[jj]=znc[jj];
		tn[jj]=tnc[jj];
		jl[jj]=jlc[jj];
		Bij[jj] = true;
		Sik[jj]=Sikc[jj];
		dSikx[jj]=dSikxc[jj];
		dSiky[jj]=dSikyc[jj];
		dSikz[jj]=dSikzc[jj];
		for (kk=0;kk<count;kk++){
			dSijkx[jj*count+kk] = dSijkxc[jj][kk];
			dSijky[jj*count+kk] = dSijkyc[jj][kk];
			dSijkz[jj*count+kk] = dSijkzc[jj][kk];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//replaced with Cholesky solution for greater speed for finding solve step. Still used to process input data.
void PairRANN::qrsolve(double *A,int m,int n,double *b, double *x_){
	double QR_[m*n];
//	char str[MAXLINE];
	double Rdiag[n];
	int i=0, j=0, k=0;
	int j_off, k_off;
	double nrm;
    // loop to copy QR from A.
	for (k=0;k<n;k++){
		k_off = k*m;
		for (i=0;i<m;i++){
			QR_[k_off+i]=A[i*n+k];
		}
	}
    for (k = 0; k < n; k++) {
       // Compute 2-norm of k-th column.
       nrm = 0.0;
       k_off = k*m;
       for (i = k; i < m; i++) {
			nrm += QR_[k_off+i]*QR_[k_off+i];
       }
       if (nrm==0.0){
    	   errorf("Jacobian is rank deficient!\n");
       }
       nrm = sqrt(nrm);
	   // Form k-th Householder vector.
	   if (QR_[k_off+k] < 0) {
		 nrm = -nrm;
 	   }
	   for (i = k; i < m; i++) {
		 QR_[k_off+i] /= nrm;
	   }
	   QR_[k_off+k] += 1.0;

	   // Apply transformation to remaining columns.
	   for (j = k+1; j < n; j++) {
		 double s = 0.0;
		 j_off = j*m;
		 for (i = k; i < m; i++) {
			s += QR_[k_off+i]*QR_[j_off+i];
		 }
		 s = -s/QR_[k_off+k];
		 for (i = k; i < m; i++) {
			QR_[j_off+i] += s*QR_[k_off+i];
		 }
	   }
       Rdiag[k] = -nrm;
    }
    //loop to find least squares
    for (int j=0;j<m;j++){
    	x_[j] = b[j];
    }
    // Compute Y = transpose(Q)*b
	for (int k = 0; k < n; k++)
	{
		k_off = k*m;
		double s = 0.0;
		for (int i = k; i < m; i++)
		{
		   s += QR_[k_off+i]*x_[i];
		}
		s = -s/QR_[k_off+k];
		for (int i = k; i < m; i++)
		{
		   x_[i] += s*QR_[k_off+i];
		}
	}
	// Solve R*X = Y;
	for (int k = n-1; k >= 0; k--)
	{
		k_off = k*m;
		x_[k] /= Rdiag[k];
		for (int i = 0; i < k; i++) {
		   x_[i] -= x_[k]*QR_[k_off+i];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
void PairRANN::chsolve(double *A,int n,double *b, double *x){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	int	nthreads=omp_get_num_threads();

	double L_[n*n]; // was L_[n][n]
	int i,j,k;
	int iXn, jXn, kXn;
	double d, s;

	// initialize L
	for (k=0;k<n*n;k++){
		L_[k]=0.0;
	}

	// Cholesky-Crout decomposition
	#pragma omp parallel default(none) shared (A,L_,n,s)
	{
	for (int j = 0; j <n; j++) {
		int jXn = j*n;
		s = 0.0;
		// #pragma omp for schedule(static) reduction(+:s)
		for (int k = 0; k < j; k++) {
			s += L_[jXn + k] * L_[jXn + k];
		}
		#pragma omp barrier
		double d = A[jXn+j] - s;
		#pragma omp single
		{
		if (d>0){
			L_[jXn + j] = sqrt(d);
		}
		}
		//// #pragma omp parallel for schedule(static) default(none) shared (A,L_,n,j,jXn)
		////#pragma omp barrier
		#pragma omp for schedule(static)
		for (int i = j+1; i <n; i++) {
			int iXn = i * n;
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += L_[iXn + k] * L_[jXn + k];
			}
			L_[iXn + j] =  (A[iXn + j] - sum) / L_[jXn + j];
		}
	}
	}
	// Solve L*
	// Forward substitution to solve L*y = b;
	// #pragma omp parallel default(none) shared (x,b,L_,n,s) private(i)
	// #pragma omp parallel
	{
	for (int k = 0; k < n; k++)
	{
		int kXn = k*n;
		s = 0.0;
		// #pragma omp parallel for default(none) reduction(+:s) schedule(static) shared (x,L_,kXn,k) private(i) if (nthreads>k)
		// #pragma omp for reduction(+:s) schedule(static)
		for (i = 0; i < k; i++) {
			s += x[i]*L_[kXn+i];
		}
		// #pragma omp single
		x[k] = (b[k] - s) / L_[kXn+k];
	}
	}
	// Backward substitution to solve L'*X = Y; omp does not work
	for (int k = n-1; k >= 0; k--)
	{
		double s = 0.0;
		for (int i = k+1; i < n; i++) {
			s += x[i]*L_[i*n+k];
		}
		x[k] = (x[k] - s)/L_[k*n+k];
	}

	//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	//printf(" - chsolve(): %f ms\n",time);

	return;
}

//writes files used for restarting and final output:
void PairRANN::write_potential_file(bool writeparameters, char *header,int iter, double reg){
	int i,j,k,l;
	char filename[strlen(potential_output_file)+10];
	if (overwritepotentials){
		sprintf(filename,"%s",potential_output_file);
	}
	else {
		sprintf(filename,"%s.%d",potential_output_file,iter);
	}
	FILE *fid = fopen(filename,"w");
	if (fid==NULL){
		errorf("Invalid parameter file name");
	}
	NNarchitecture *net_out = new NNarchitecture[nelementsp];
	if (normalizeinput){
		unnormalize_net(net_out);
	}
	else {
		copy_network(net,net_out);
	}
	fprintf(fid,"#");
	fprintf(fid,header);
	//atomtypes section
	fprintf(fid,"atomtypes:\n");
	for (i=0;i<nelements;i++){
		fprintf(fid,"%s ",elements[i]);
	}
	fprintf(fid,"\n");
	//mass section
	for (i=0;i<nelements;i++){
		fprintf(fid,"mass:%s:\n",elements[i]);
		fprintf(fid,"%f\n",mass[i]);
	}
	//fingerprints per element section
	for (i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			fprintf(fid,"fingerprintsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",fingerprintperelement[i]);
		}
	}
	//fingerprints section:
	for (i=0;i<nelementsp;i++){
		bool printheader = true;
		for (j=0;j<fingerprintperelement[i];j++){
			if (printheader){
				fprintf(fid,"fingerprints:");
				fprintf(fid,"%s",elementsp[fingerprints[i][j]->atomtypes[0]]);
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					fprintf(fid,"_%s",elementsp[fingerprints[i][j]->atomtypes[k]]);
				}
				fprintf(fid,":\n");
			}
			else {fprintf(fid,"\t");}
			fprintf(fid,"%s_%d",fingerprints[i][j]->style,fingerprints[i][j]->id);
			printheader = true;
			if (j<fingerprintperelement[i]-1 && fingerprints[i][j]->n_body_type == fingerprints[i][j+1]->n_body_type){
				printheader = false;
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					if (fingerprints[i][j]->atomtypes[k]!=fingerprints[i][j+1]->atomtypes[k]){
						printheader = true;
						fprintf(fid,"\n");
						break;
					}
				}
			}
			else fprintf(fid,"\n");
		}
	}
	//fingerprint contants section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<fingerprintperelement[i];j++){
			fingerprints[i][j]->write_values(fid);
		}
	}
	//screening section
	for (i=0;i<nelements;i++){
		for (j=0;j<nelements;j++){
			for (k=0;k<nelements;k++){
				fprintf(fid,"screening:%s_%s_%s:Cmax:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_max[i*nelements*nelements+j*nelements+k]);
				fprintf(fid,"screening:%s_%s_%s:Cmin:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_min[i*nelements*nelements+j*nelements+k]);
			}
		}
	}
	//network layers section:
	for (i=0;i<nelementsp;i++){
		if (net_out[i].layers>0){
			fprintf(fid,"networklayers:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",net_out[i].layers);
		}
	}
	//layer size section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers;j++){
			fprintf(fid,"layersize:%s:%d:\n",elementsp[i],j);
			fprintf(fid,"%d\n",net_out[i].dimensions[j]);
		}
	}
	//bundles section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			fprintf(fid,"bundles:%s:%d:\n",elements[i],j);
			fprintf(fid,"%d\n",net_out[i].bundles[j]);
		}
	}
	//bundle id section
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			for (int i1=0;i1<net_out[i].bundles[j];i1++){
				if (net_out[i].identitybundle[j][i1]){
					fprintf(fid,"bundleid:%s:%d:%d:\n",elements[i],j,i1);
					fprintf(fid,"1\n");
				}
			}
		}
	}
	//bundle input section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			for (int i1=0;i1<net_out[i].bundles[j];i1++){
				fprintf(fid,"bundleinput:%s:%d:%d:\n",elements[i],j,i1);
				for (k=0;k<net_out[i].bundleinputsize[j][i1];k++){
					fprintf(fid,"%d ",net_out[i].bundleinput[j][i1][k]);
				}
				fprintf(fid,"\n");
			}
		}
	}
	//bundle output section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			for (int i1=0;i1<net_out[i].bundles[j];i1++){
				fprintf(fid,"bundleoutput:%s:%d:%d:\n",elements[i],j,i1);
				for (k=0;k<net_out[i].bundleoutputsize[j][i1];k++){
					fprintf(fid,"%d ",net_out[i].bundleoutput[j][i1][k]);
				}
				fprintf(fid,"\n");
			}
		}
	}
	//weight section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			for (int i1=0;i1<net_out[i].bundles[j];i1++){
				if (net_out[i].identitybundle[j][i1])continue;
				fprintf(fid,"weight:%s:%d:%d:\n",elementsp[i],j,i1);
				for (k=0;k<net_out[i].bundleoutputsize[j][i1];k++){
					for (l=0;l<net_out[i].bundleinputsize[j][i1];l++){
						fprintf(fid,"%.15e\t",net_out[i].bundleW[j][i1][k*net_out[i].bundleinputsize[j][i1]+l]);
					}
					fprintf(fid,"\n");
				}
			}
		}
	}
	//bias section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			for (int i1=0;i1<net_out[i].bundles[j];i1++){
				if (net_out[i].identitybundle[j][i1])continue;
				fprintf(fid,"bias:%s:%d:%d:\n",elementsp[i],j,i1);
				for (k=0;k<net_out[i].bundleoutputsize[j][i1];k++){
					fprintf(fid,"%.15e\n",net_out[i].bundleB[j][i1][k]);
				}
			}
		}
	}
	//activation section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<net_out[i].layers-1;j++){
			bool allsame = true;
			for (k=1;k<net_out[i].dimensions[j+1];k++){
				if (strcmp(activation[i][j][k]->style,activation[i][j][0]->style)!=0){
					allsame = false;
					break;
				}
			}
			if (!allsame){
				for (k=0;k<net_out[i].dimensions[j+1];k++){
					fprintf(fid,"activationfunctions:%s:%d:%d:\n",elementsp[i],j,k);
					fprintf(fid,"%s\n",activation[i][j][k]->style);
				}
			}
			else {
				fprintf(fid,"activationfunctions:%s:%d:\n",elementsp[i],j);
				fprintf(fid,"%s\n",activation[i][j][0]->style);
			}
		}
	}
	//state equation per element section
	for (i=0;i<nelementsp;i++){
		if (stateequationperelement[i]>0){
			fprintf(fid,"stateequationsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",stateequationperelement[i]);
		}
	}
	//state equations section:
	for (i=0;i<nelementsp;i++){
		bool printheader = true;
		for (j=0;j<stateequationperelement[i];j++){
			if (printheader){
				fprintf(fid,"stateequations:");
				fprintf(fid,"%s",elementsp[state[i][j]->atomtypes[0]]);
				for (k=1;k<state[i][j]->n_body_type;k++){
					fprintf(fid,"_%s",elementsp[state[i][j]->atomtypes[k]]);
				}
				fprintf(fid,":\n");
			}
			else {fprintf(fid,"\t");}
			fprintf(fid,"%s_%d",state[i][j]->style,state[i][j]->id);
			printheader = true;
			if (j<stateequationperelement[i]-1 && state[i][j]->n_body_type == state[i][j+1]->n_body_type){
				printheader = false;
				for (k=1;k<state[i][j]->n_body_type;k++){
					if (state[i][j]->atomtypes[k]!=state[i][j+1]->atomtypes[k]){
						printheader = true;
						fprintf(fid,"\n");
						break;
					}
				}
			}
			else fprintf(fid,"\n");
		}
	}
	//state equations contants section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<stateequationperelement[i];j++){
			state[i][j]->write_values(fid);
		}
	}
	//calibration parameters section
	if (writeparameters){
		fprintf(fid,"calibrationparameters:algorithm:\n");
		fprintf(fid,"%s\n",algorithm);
		fprintf(fid,"calibrationparameters:dumpdirectory:\n");
		fprintf(fid,"%s\n",dump_directory);
		fprintf(fid,"calibrationparameters:doforces:\n");
		fprintf(fid,"%d\n",doforces);
		fprintf(fid,"calibrationparameters:normalizeinput:\n");
		fprintf(fid,"%d\n",normalizeinput);
		fprintf(fid,"calibrationparameters:tolerance:\n");
		fprintf(fid,"%.10e\n",tolerance);
		fprintf(fid,"calibrationparameters:regularizer:\n");
		fprintf(fid,"%.10e\n",reg);
		fprintf(fid,"calibrationparameters:logfile:\n");
		fprintf(fid,"%s\n",log_file);
		fprintf(fid,"calibrationparameters:potentialoutputfile:\n");
		fprintf(fid,"%s\n",potential_output_file);
		fprintf(fid,"calibrationparameters:potentialoutputfreq:\n");
		fprintf(fid,"%d\n",potential_output_freq);
		fprintf(fid,"calibrationparameters:maxepochs:\n");
		fprintf(fid,"%d\n",max_epochs);
		for (i=0;i<nelements;i++){
			for (j=0;j<net_out[i].layers-1;j++){
				for (int k=0;k<net_out[i].bundles[j];k++){
					if (net_out[i].identitybundle[j][k])continue;
					bool anyfrozen = false;
					for (int l=0;l<net_out[i].bundleoutputsize[j][k]*net_out[i].bundleinputsize[j][k];l++){
						if (net_out[i].freezeW[j][k][l]){
							anyfrozen = true;
							break;
						}
					}
					if (anyfrozen){
						fprintf(fid,"calibrationparameters:freezeW:%d:%d\n",j,k);
						for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
							for (int m=0;m<net_out[i].bundleinputsize[j][k];m++){
								fprintf(fid,"%d ",net_out[i].freezeW[j][k][l*net_out[i].bundleoutputsize[j][k]+m]);
							}
							fprintf(fid,"\n");
						}
					}
					anyfrozen = false;
					for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
						if (net_out[i].freezeB[j][k][l]){
							anyfrozen = true;
							break;
						}
					}
					if (anyfrozen){
						fprintf(fid,"calibrationparameters:freezeB:%d:%d\n",j,k);
						for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
							fprintf(fid,"%d\n",net_out[i].freezeB[j][k][l]);
						}
					}
				}
			}
		}
		fprintf(fid,"calibrationparameters:validation:\n");
		fprintf(fid,"%f\n",validation);
		fprintf(fid,"calibrationparameters:overwritepotentials:\n");
		fprintf(fid,"%d\n",overwritepotentials);
		fprintf(fid,"calibrationparameters:debug1freq:\n");
		fprintf(fid,"%d\n",debug_level1_freq);
		fprintf(fid,"calibrationparameters:debug2freq:\n");
		fprintf(fid,"%d\n",debug_level2_freq);
		fprintf(fid,"calibrationparameters:debug3freq:\n");
		fprintf(fid,"%d\n",debug_level3_freq);
		fprintf(fid,"calibrationparameters:debug4freq:\n");
		fprintf(fid,"%d\n",debug_level4_freq);
		fprintf(fid,"calibrationparameters:debug5freq:\n");
		fprintf(fid,"%d\n",debug_level5_freq);
		fprintf(fid,"calibrationparameters:debug6freq:\n");
		fprintf(fid,"%d\n",debug_level6_freq);
		fprintf(fid,"calibrationparameters:adaptiveregularizer:\n");
		fprintf(fid,"%d\n",adaptive_regularizer);
		fprintf(fid,"calibrationparameters:lambdainitial:\n");
		fprintf(fid,"%f\n",lambda_initial);
		fprintf(fid,"calibrationparameters:lambdaincrease:\n");
		fprintf(fid,"%f\n",lambda_increase);
		fprintf(fid,"calibrationparameters:lambdareduce:\n");
		fprintf(fid,"%f\n",lambda_reduce);
		fprintf(fid,"calibrationparameters:seed:\n");
		fprintf(fid,"%d\n",seed);
	}
	fclose(fid);
	delete [] net_out;
}




void PairRANN::screen(double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz, bool *Bij, int ii,int sid,double *xn,double *yn,double *zn,int *tn,int jnum)
{
	//#pragma omp parallel
	{
	//see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
	int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
	double Sijk,Cijk,Cn,Cd,Dij,Dik,Djk,C,dfc,dC,**x;
	PairRANN::Simulation *sim = &sims[sid];
//	x = sim->x;
	double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
	i = sim->ilist[ii];
	itype = map[sim->type[i]];
//	jnum = sim->numneigh[i];
//	jlist = sim->firstneigh[i];
//	xtmp = x[i][0];
//	ytmp = x[i][1];
//	ztmp = x[i][2];
	for (int jj=0;jj<jnum;jj++){
		Sik[jj]=1;
		Bij[jj]=true;
		dSikx[jj]=0;
		dSiky[jj]=0;
		dSikz[jj]=0;
	}
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkx[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijky[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkz[jj*jnum+kk]=0;

	//#pragma omp for schedule (dynamic)for collapse(2)
	for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
		if (Bij[kk]==false){continue;}
//		k = jlist[kk];
//		k &= NEIGHMASK;
//		ktype = map[sim->type[k]];
		ktype = tn[kk];
//		delx2 = xtmp - x[k][0];
//		dely2 = ytmp - x[k][1];
//		delz2 = ztmp - x[k][2];
		delx2 = xn[kk];
		dely2 = yn[kk];
		delz2 = zn[kk];
		rik = delx2*delx2+dely2*dely2+delz2*delz2;
		if (rik>cutmax*cutmax){
			Bij[kk]= false;
			continue;
		}
		for (jj=0;jj<jnum;jj++){
			if (jj==kk){continue;}
			if (Bij[jj]==false){continue;}
//			j = jlist[jj];
//			j &= NEIGHMASK;
//			jtype = map[sim->type[j]];
//			delx = xtmp - x[j][0];
//			dely = ytmp - x[j][1];
//			delz = ztmp - x[j][2];
			jtype = tn[jj];
			delx = xn[jj];
			dely = yn[jj];
			delz = zn[jj];
			rij = delx*delx+dely*dely+delz*delz;
			if (rij>cutmax*cutmax){
				Bij[jj] = false;
				continue;
			}
//			delx3 = x[j][0]-x[k][0];
//			dely3 = x[j][1]-x[k][1];
//			delz3 = x[j][2]-x[k][2];
			delx3 = delx2-delx;
			dely3 = dely2-dely;
			delz3 = delz2-delz;
			rjk = delx3*delx3+dely3*dely3+delz3*delz3;
			if (rik+rjk-rij<1e-14){continue;}//bond angle > 90 degrees
			if (rik+rij-rjk<1e-14){continue;}//bond angle > 90 degrees
			double Cmax = screening_max[itype*nelements*nelements+jtype*nelements+ktype];
			double Cmin = screening_min[itype*nelements*nelements+jtype*nelements+ktype];
			double temp1 = rij-rik+rjk;
			Cn = temp1*temp1-4*rij*rjk;
			//Cn = (rij-rik+rjk)*(rij-rik+rjk)-4*rij*rjk;
			temp1 = rij-rjk;
			Cd = temp1*temp1-rik*rik;
			//Cd = (rij-rjk)*(rij-rjk)-rik*rik;
			Cijk = Cn/Cd;
			//Cijk = 1+2*(rik*rij+rik*rjk-rik*rik)/(rik*rik-(rij-rjk)*(rij-rjk));
			C = (Cijk-Cmin)/(Cmax-Cmin);
			if (C>=1){continue;}
			else if (C<=0){
				Bij[kk]=false;
				break;
			}
			dC = Cmax-Cmin;
			dC *= dC;
			dC *= dC;
			temp1 = 1-C;
			temp1 *= temp1;
			temp1 *= temp1;
			Sijk = 1-temp1;
			Sijk *= Sijk;
			Dij = 4*rik*(Cn+4*rjk*(rij+rik-rjk))/Cd/Cd;
			Dik = -4*(rij*Cn+rjk*Cn+8*rij*rik*rjk)/Cd/Cd;
			Djk = 4*rik*(Cn+4*rij*(rik-rij+rjk))/Cd/Cd;
			temp1 = Cijk-Cmax;
			double temp2 = temp1*temp1;
			dfc = 8*temp1*temp2/(temp2*temp2-dC);
			Sik[kk] *= Sijk;
			dSijkx[kk*jnum+jj] = dfc*(delx*Dij-delx3*Djk);
			dSikx[kk] += dfc*(delx2*Dik+delx3*Djk);
			dSijky[kk*jnum+jj] = dfc*(dely*Dij-dely3*Djk);
			dSiky[kk] += dfc*(dely2*Dik+dely3*Djk);
			dSijkz[kk*jnum+jj] = dfc*(delz*Dij-delz3*Djk);
			dSikz[kk] += dfc*(delz2*Dik+delz3*Djk);
		}
	}
	}
}

//treats # as starting a comment to be ignored.
int PairRANN::count_words(char *line){
	return count_words(line,": ,\t_\n");
}

int PairRANN::count_words(char *line,char *delimiter){
	int n = strlen(line) + 1;
	char copy[n];
	strncpy(copy,line,n);
	char *ptr;
	if ((ptr = strchr(copy,'#'))) *ptr = '\0';
	if (strtok(copy,delimiter) == NULL) {
		return 0;
	}
	n=1;
	while ((strtok(NULL,delimiter))) n++;
	return n;
}

void PairRANN::errorf(const std::string &file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file,line,message);
	exit(1);
}

void PairRANN::errorf(char *file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file,line,message);
	exit(1);
}

void PairRANN::errorf(const char *message){
	//see about adding message to log file
	std::cout<<message;
	std::cout<<"\n";
	exit(1);
}


int PairRANN::factorial(int n) {
   if ((n==0)||(n==1))
      return 1;
   else
      return n*factorial(n-1);
}

std::vector<std::string> PairRANN::tokenmaker(std::string line,std::string delimiter){
	int nwords = count_words(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	char **words=new char *[nwords+1];
	nwords = 0;
	words[nwords++]=strtok(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	while ((words[nwords++] = strtok(NULL,const_cast<char *>(delimiter.c_str())))) continue;
	nwords--;
	std::vector<std::string> linev;
	for (int i=0;i<nwords;i++){
		linev.emplace_back(words[i]);
	}
	delete [] words;
	return linev;
}