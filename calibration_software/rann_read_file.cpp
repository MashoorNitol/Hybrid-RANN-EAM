#include "pair_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"
using namespace LAMMPS_NS;


void PairRANN::read_file(char *filename)
{
  FILE *fp;
  int eof = 0;
  std::string line,line1;
  const int longline = 4096;
  int linenum=0;
  char linetemp[longline];
  std::string strtemp;
  char *ptr;
  std::vector<std::string> linev,line1v;
  fp = utils::open_potential(filename,lmp,nullptr);
  if (fp == nullptr) {errorf(FLERR,"Cannot open RANN potential file");}
  ptr=fgets(linetemp,longline,fp);
  linenum++;
  strtemp=utils::trim_comment(linetemp);
  while (strtemp.empty()) {
          ptr=fgets(linetemp,longline,fp);
          strtemp=utils::trim_comment(linetemp);
          linenum++;
  }
  line=strtemp;
  while (eof == 0) {
    ptr=fgets(linetemp,longline,fp);
    linenum++;
    if (ptr == nullptr) {
      fclose(fp);
      if (check_potential()) {
        errorf(FLERR,"Invalid syntax in potential file, values are inconsistent or missing");
      }
      else {
        eof = 1;
        break;
      }
    }
    strtemp=utils::trim_comment(linetemp);
    while (strtemp.empty()) {
        ptr=fgets(linetemp,longline,fp);
        strtemp=utils::trim_comment(linetemp);
        linenum++;
    }
    line1=linetemp;
    linev = tokenmaker(line,": ,\t_\n");
    line1v = tokenmaker(line1,": ,\t_\n");
    if (linev[0]=="atomtypes") read_atom_types(line1v,filename,linenum);
    else if (linev[0]=="mass") read_mass(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprintsperelement") read_fpe(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprints") read_fingerprints(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprintconstants") read_fingerprint_constants(linev,line1v,filename,linenum);
    else if (linev[0]=="networklayers") read_network_layers(linev,line1v,filename,linenum);
    else if (linev[0]=="layersize") read_layer_size(linev,line1v,filename,linenum);
    else if (linev[0]=="weight") read_weight(linev,line1v,fp,filename,&linenum);
    else if (linev[0]=="bias") read_bias(linev,line1v,fp,filename,&linenum);
    else if (linev[0]=="activationfunctions") read_activation_functions(linev,line1v,filename,linenum);
    else if (linev[0]=="screening") read_screening(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequationsperelement") read_eospe(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequations") read_eos(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequationconstants") read_eos_constants(linev,line1v,filename,linenum);
    else if (linev[0]=="bundles")read_bundles(linev,line1v,filename,linenum);
    else if (linev[0]=="bundleinput")read_bundle_input(linev,line1v,filename,linenum);
    else if (linev[0]=="bundleoutput")read_bundle_output(linev,line1v,filename,linenum);
    else if (linev[0]=="bundleid")read_bundle_id(linev,line1v,filename,linenum);
    else if (linev[0]=="calibrationparameters") {
        if (~is_lammps)read_parameters(linev,line1v,fp,filename,&linenum);
    }
    else errorf(filename,linenum-1,"Could not understand file syntax: unknown keyword");
    ptr=fgets(linetemp,longline,fp);
    linenum++;
    strtemp=utils::trim_comment(linetemp);
    while (strtemp.empty()) {
        ptr=fgets(linetemp,longline,fp);
        strtemp=utils::trim_comment(linetemp);
        linenum++;
    }
    if (ptr == nullptr) {
      if (check_potential()) {
        errorf(FLERR,"Invalid syntax in potential file, values are inconsistent or missing");
      }
      else {
        eof = 1;
        break;
      }
    }
    line=linetemp;
  }
}

void PairRANN::read_atom_types(std::vector<std::string> line,char *filename,int linenum) {
  int nwords = line.size();
  if (nwords < 1) errorf(filename,linenum,"Incorrect syntax for atom types");
  nelements = nwords;
  line.emplace_back("all");
  allocate(line);
}

void PairRANN::read_mass(const std::vector<std::string> &line1, const std::vector<std::string> &line2, const char *filename,int linenum) {
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before mass in potential file.");
  for (int i=0;i<nelements;i++) {
    if (line1[1].compare(elements[i])==0) {
      mass[i]=utils::numeric(filename,linenum,line2[0],true,lmp);
      return;
    }
  }
  errorf(filename,linenum-1,"mass element not found in atom types.");
}

void PairRANN::read_fpe(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints per element in potential file.");
  for (i=0;i<nelementsp;i++) {
    if (line[1].compare(elementsp[i])==0) {
      fingerprintperelement[i] = utils::inumeric(filename,linenum,line1[0],true,lmp);
      fingerprints[i] = new RANN::Fingerprint *[fingerprintperelement[i]];
      for (int j=0;j<fingerprintperelement[i];j++) {
        fingerprints[i][j]=new RANN::Fingerprint(this);
      }
      return;
    }
  }
  errorf(filename,linenum-1,"fingerprint-per-element element not found in atom types");
}

void PairRANN::read_fingerprints(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int nwords1,nwords,i,j,k,i1,*atomtypes;
  bool found;
  nwords1 = line1.size();
  nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  atomtypes = new int[nwords-1];
  for (i=1;i<nwords;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  k = 0;
  if (fingerprintperelement[i]==-1) {errorf(filename,linenum-1,"fingerprint per element must be defined before fingerprints");}
  while (k<nwords1) {
    i1 = fingerprintcount[i];
    if (i1>=fingerprintperelement[i]) {errorf(filename,linenum,"more fingerprints defined than fingerprint per element");}
    delete fingerprints[i][i1];
    fingerprints[i][i1] = create_fingerprint(line1[k].c_str());
    if (fingerprints[i][i1]->n_body_type!=nwords-1) {errorf(filename,linenum,"invalid fingerprint for element combination");}
    k++;
    fingerprints[i][i1]->init(atomtypes,utils::inumeric(filename,linenum,line1[k++],true,lmp));
    fingerprintcount[i]++;
  }
  delete[] atomtypes;
}

void PairRANN::read_fingerprint_constants(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,i1,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  int n_body_type = nwords-4;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  found = false;
  for (k=0;k<fingerprintperelement[i];k++) {
    if (fingerprints[i][k]->empty) {continue;}
    if (n_body_type!=fingerprints[i][k]->n_body_type) {continue;}
    for (j=0;j<n_body_type;j++) {
      if (fingerprints[i][k]->atomtypes[j]!=atomtypes[j]) {break;}
      if (j==n_body_type-1) {
        if (line[nwords-3].compare(fingerprints[i][k]->style)==0 && utils::inumeric(filename,linenum,line[nwords-2],true,lmp)==fingerprints[i][k]->id) {
          found=true;
          i1 = k;
          break;
        }
      }
    }
    if (found) {break;}
  }
  if (!found) {errorf(filename,linenum-1,"cannot define constants for unknown fingerprint");}
  fingerprints[i][i1]->fullydefined=fingerprints[i][i1]->parse_values(line[nwords-1],line1);
  delete[] atomtypes;
}

void PairRANN::read_network_layers(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j;
  if (nelements == -1)errorf(FLERR,"atom types must be defined before network layers in potential file.");
  for (i=0;i<nelements;i++){
    if (line[1].compare(elements[i])==0){
      net[i].layers = utils::inumeric(filename,linenum,line1[0],true,lmp);
      if (net[i].layers < 1)errorf(filename,linenum,"invalid number of network layers");
      weightdefined[i] = new bool *[net[i].layers];
      biasdefined[i] = new bool *[net[i].layers];
      dimensiondefined[i] = new bool [net[i].layers];
      bundle_inputdefined[i] = new bool *[net[i].layers];
      bundle_outputdefined[i] = new bool *[net[i].layers];
      net[i].dimensions = new int [net[i].layers];
      net[i].bundles = new int[net[i].layers];
      net[i].identitybundle = new bool *[net[i].layers];
      net[i].bundleinputsize = new int *[net[i].layers];
      net[i].bundleoutputsize = new int *[net[i].layers];
      net[i].bundleinput = new int **[net[i].layers];
      net[i].bundleoutput = new int **[net[i].layers];
      net[i].bundleW = new double **[net[i].layers];
      net[i].bundleB = new double **[net[i].layers];
      net[i].freezeW = new bool **[net[i].layers];
      net[i].freezeB = new bool **[net[i].layers];
      for (j=0;j<net[i].layers;j++){
        net[i].dimensions[j]=0;
        net[i].bundles[j] = 1;
        net[i].bundleinputsize[j] = new int [net[i].bundles[j]];
        net[i].bundleoutputsize[j] = new int [net[i].bundles[j]];
        net[i].bundleinput[j] = new int *[net[i].bundles[j]];
        net[i].bundleoutput[j] = new int *[net[i].bundles[j]];
        net[i].bundleW[j] = new double *[net[i].bundles[j]];
        net[i].bundleB[j] = new double *[net[i].bundles[j]];
        net[i].freezeW[j] = new bool *[net[i].bundles[j]];
        net[i].freezeB[j] = new bool *[net[i].bundles[j]];
        net[i].identitybundle[j] = new bool[net[i].bundles[j]];
        weightdefined[i][j] = new bool[net[i].bundles[j]];
        biasdefined[i][j] = new bool[net[i].bundles[j]];
        dimensiondefined[i][j]=false;
        bundle_inputdefined[i][j]=new bool [net[i].bundles[j]];
        bundle_outputdefined[i][j]=new bool [net[i].bundles[j]];
        for (int k=0;k<net[i].bundles[j];k++){
          weightdefined[i][j][k]=false;
          biasdefined[i][j][k]=false;
          bundle_inputdefined[i][j][k]=false;
          bundle_outputdefined[i][j][k]=false;
          net[i].identitybundle[j][k]=false;
        }
      }
      activation[i]=new RANN::Activation** [net[i].layers-1];
	  for (int j=0;j<net[i].layers-1;j++) {
        activation[i][j]= new RANN::Activation*;
      }
      return;
    }
  }
  errorf(filename,linenum-1,"network layers element not found in atom types");
}

void PairRANN::read_layer_size(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  for (i=0;i<nelements;i++) {
    if (line[1].compare(elements[i])==0) {
      if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
      int j = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (j>=net[i].layers || j<0) {errorf(filename,linenum,"invalid layer in layer size definition");};
      net[i].dimensions[j]= utils::inumeric(filename,linenum,line1[0],true,lmp);
      dimensiondefined[i][j]=true;
      if (j>0){
        activation[i][j-1] = new RANN::Activation*[net[i].dimensions[j]];
        for (int k=0;k<net[i].dimensions[j];k++){
          activation[i][j-1][k] = new RANN::Activation(this);
        }
      }
      return;
    }
  }
  errorf(filename,linenum-1,"layer size element not found in atom types");
}

void PairRANN::read_bundles(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
  int i;
  for (i=0;i<nelements;i++){
    if (line[1].compare(elements[i])==0){
      if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
      int j = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundles definition");};
      net[i].bundles[j]= utils::inumeric(filename,linenum,line1[0],true,lmp);
      delete [] net[i].bundleinputsize[j];
      delete [] net[i].bundleoutputsize[j];
      delete [] net[i].bundleinput[j];
      delete [] net[i].bundleoutput[j];
      delete [] net[i].bundleW[j];
      delete [] net[i].bundleB[j];
      delete [] net[i].freezeW[j];
      delete [] net[i].freezeB[j];
      delete [] net[i].identitybundle[j];
      delete [] weightdefined[i][j];
      delete [] biasdefined[i][j];
      delete [] bundle_inputdefined[i][j];
      delete [] bundle_outputdefined[i][j];
      net[i].bundleinputsize[j] = new int [net[i].bundles[j]];
      net[i].bundleoutputsize[j] = new int [net[i].bundles[j]];
      net[i].bundleinput[j] = new int *[net[i].bundles[j]];
      net[i].bundleoutput[j] = new int *[net[i].bundles[j]];
      net[i].bundleW[j] = new double *[net[i].bundles[j]];
      net[i].bundleB[j] = new double *[net[i].bundles[j]];
      net[i].freezeW[j] = new bool *[net[i].bundles[j]];
      net[i].freezeB[j] = new bool *[net[i].bundles[j]];
      net[i].identitybundle[j] = new bool[net[i].bundles[j]];
      weightdefined[i][j] = new bool[net[i].bundles[j]];
      biasdefined[i][j] = new bool[net[i].bundles[j]];
      bundle_inputdefined[i][j] = new bool[net[i].bundles[j]];
      bundle_outputdefined[i][j] = new bool[net[i].bundles[j]];
      for (int k=0;k<net[i].bundles[j];k++){
        weightdefined[i][j][k]=false;
        biasdefined[i][j][k]=false;
        bundle_inputdefined[i][j][k]=false;
        bundle_outputdefined[i][j][k]=false;
        net[i].identitybundle[j][k]=false;
      }
      return;
    }
  }
  errorf(FLERR,"bundle element not found in atom types");
}

void PairRANN::read_bundle_id(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
  int i;
  for (i=0;i<nelements;i++){
    if (line[1].compare(elements[i])==0){
      if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
      int j = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
      int b = utils::inumeric(filename,linenum,line[3],true,lmp);
      if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
      int id =utils::inumeric(filename,linenum,line1[0],true,lmp);
      if (id!= 1 && id != 0)errorf(filename,linenum-1,"bundle id must be 1 or 0\n");
      net[i].identitybundle[j][b]=id;
      return;
    }
  }
  errorf(filename,linenum-1,"bundle id element not found in atom types");
}

void PairRANN::read_bundle_input(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
  int i,nwords;
  for (i=0;i<nelements;i++){
    if (line[1].compare(elements[i])==0){
      if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
      int j = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
      int b = utils::inumeric(filename,linenum,line[3],true,lmp);
      if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
      nwords = line1.size();
      net[i].bundleinputsize[j][b] = nwords;
      net[i].bundleinput[j][b] = new int [nwords];
      for (int k=0;k<nwords;k++){
        net[i].bundleinput[j][b][k]=utils::inumeric(filename,linenum,line1[k],true,lmp);
      }
      bundle_inputdefined[i][j][b]=true;
      return;
    }
  }
  errorf(filename,linenum-1,"bundle input element not found in atom types");
}

void PairRANN::read_bundle_output(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
  int i,nwords;
  for (i=0;i<nelements;i++){
    if (line[1].compare(elements[i])==0){
      if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
      int j = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
      int b = utils::inumeric(filename,linenum,line[3],true,lmp);
      if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
      nwords = line1.size();
      net[i].bundleoutputsize[j][b] = nwords;
      net[i].bundleoutput[j][b] = new int [nwords];
      for (int k=0;k<nwords;k++){
        net[i].bundleoutput[j][b][k]=utils::inumeric(filename,linenum,line1[k],true,lmp);
      }
      bundle_outputdefined[i][j][b]=true;
      return;
    }
  }
  errorf(filename,linenum-1,"bundle output element not found in atom types");
}

void PairRANN::read_weight(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,k,l,b,ins,ops,nwords;
  char *ptr;
  char **words1;
  int longline = 4096;
  char linetemp [longline];
  nwords = line.size();
  if (nwords == 3){b=0;}
  else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,lmp);}
  for (l=0;l<nelements;l++){
    if (line[1].compare(elements[l])==0){
      if (net[l].layers==0)errorf(filename,*linenum-1,"networklayers must be defined before weights.");
      i=utils::inumeric(filename,*linenum,line[2],true,lmp);
      if (i>=net[l].layers || i<0)errorf(filename,*linenum-1,"invalid weight layer");
      if (dimensiondefined[l][i]==false || dimensiondefined[l][i+1]==false) errorf(FLERR,"network layer sizes must be defined before corresponding weight");
      if (bundle_inputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle inputs must be defined before weights");
      if (bundle_outputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle outputs must be defined before weights");
      if (net[l].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define weights for an identity bundle");
      if (bundle_inputdefined[l][i][b]==false){ins = net[l].dimensions[i];}
      else {ins = net[l].bundleinputsize[i][b];}
      if (bundle_outputdefined[l][i][b]==false){ops = net[l].dimensions[i+1];}
      else {ops = net[l].bundleoutputsize[i][b];}
      net[l].bundleW[i][b] = new double [ins*ops];
      net[l].freezeW[i][b] = new bool [ins*ops];
      for (j=0;j<ins*ops;j++){net[l].freezeW[i][b][j]=0;}
      weightdefined[l][i][b] = true;
	    nwords = line1.size();
      if (nwords != ins)errorf(filename,*linenum-1,"invalid weights per line");
      for (k=0;k<ins;k++){
        net[l].bundleW[i][b][k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
      }
      for (j=1;j<ops;j++){
        ptr = fgets(linetemp,longline,fp);
        (linenum)++;
        line1 = tokenmaker(linetemp,": ,\t_\n");
        if (ptr==nullptr)errorf(filename,*linenum-1,"unexpected end of potential file!");
        nwords = line1.size();
        if (nwords != net[l].dimensions[i])errorf(filename,*linenum-1,"invalid weights per line");
        for (k=0;k<net[l].dimensions[i];k++) {
          net[l].bundleW[i][b][j*ins+k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
        }
      }
      return;
    }
  }
  errorf(FLERR,"weight element not found in atom types");
}

void PairRANN::read_bias(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,l,b,ops;
  char *ptr;
  int longline = 1024;
  int nwords = line.size();
  char linetemp [longline];
    if (nwords == 3){b=0;}
    else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,lmp);}
  for (l=0;l<nelements;l++){
    if (line[1].compare(elements[l])==0){
      if (net[l].layers==0)errorf(filename,*linenum-1,"networklayers must be defined before biases.");
      i=utils::inumeric(filename,*linenum,line[2],true,lmp);
      if (i>=net[l].layers || i<0)errorf(filename,*linenum-1,"invalid bias layer");
      if (dimensiondefined[l][i]==false) errorf(filename,*linenum-1,"network layer sizes must be defined before corresponding bias");
      if (bundle_outputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle outputs must be defined before bias");
      if (net[l].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define bias for an identity bundle");
      if (bundle_outputdefined[l][i][b]==false){ops=net[l].dimensions[i+1];}
      else {ops = net[l].bundleoutputsize[i][b];}
      net[l].bundleB[i][b] = new double [ops];
      net[l].freezeB[i][b] = new bool[ops];
      for (j=0;j<ops;j++){net[l].freezeB[i][b][j]=0;}
      biasdefined[l][i][b] = true;
      net[l].bundleB[i][b][0] = utils::numeric(filename,*linenum,line1[0],true,lmp);
      for (j=1;j<ops;j++){
		    ptr=fgets(linetemp,longline,fp);
        if (ptr==nullptr)errorf(filename,*linenum,"unexpected end of potential file!");
        (linenum)++;
        line1 = tokenmaker(linetemp,": ,\t_\n");
        net[l].bundleB[i][b][j] = utils::numeric(filename,*linenum,line1[0],true,lmp);
      }
      return;
    }
  }
  errorf(filename,*linenum-1,"bias element not found in atom types");
}

void PairRANN::read_activation_functions(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
  int i,l;
  int nwords = line.size();
  for (l=0;l<nelements;l++){
    if (line[1].compare(elements[l])==0){
      i = utils::inumeric(filename,linenum,line[2],true,lmp);
      if (i>=net[l].layers || i<0)errorf(filename,linenum-1,"invalid activation layer");
      if (dimensiondefined[l][i+1]==false) errorf(filename,linenum-1,"network layer sizes must be defined before corresponding activation");
      if (nwords==3){
        for (int j = 0;j<net[l].dimensions[i+1];j++){
          delete activation[l][i][j];
          activation[l][i][j]=create_activation(line1[0].c_str());
        }
	    }
	    else if (nwords>3){
		    int j = utils::inumeric(filename,linenum,line[3],true,lmp);
		    delete activation[l][i][j];
		    activation[l][i][j]=create_activation(line1[0].c_str());
	    }
      return;
    }
  }
  errorf(filename,linenum-1,"activation function element not found in atom types");
}

void PairRANN::read_eospe(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before equations of state per element in potential file.");
  for (i=0;i<nelementsp;i++) {
    if (line[1].compare(elementsp[i])==0) {
      stateequationperelement[i] = utils::inumeric(filename,linenum,line1[0],true,lmp);
      state[i] = new RANN::State *[stateequationperelement[i]];
      for (int j=0;j<stateequationperelement[i];j++) {
        state[i][j]=new RANN::State(this);
      }
      return;
    }
  }
  errorf(filename,linenum-1,"state equations-per-element element not found in atom types");
}

void PairRANN::read_eos(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int nwords1,nwords,i,j,k,i1,*atomtypes;
  bool found;
  nwords1 = line1.size();
  nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before state equations in potential file.");
  atomtypes = new int[nwords-1];
  for (i=1;i<nwords;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"state equation element not found in atom types");}
  }
  i = atomtypes[0];
  k = 0;
  if (stateequationperelement[i]==-1) {errorf(filename,linenum-1,"eos per element must be defined before fingerprints");}
  while (k<nwords1) {
    i1 = stateequationcount[i];
    if (i1>=stateequationperelement[i]) {errorf(filename,linenum,"more state equations defined than eos per element");}
    delete state[i][i1];
    state[i][i1] = create_state(line1[k].c_str());
    if (state[i][i1]->n_body_type!=nwords-1) {errorf(filename,linenum,"invalid state equation for element combination");}
    k++;
    state[i][i1]->init(atomtypes,utils::inumeric(filename,linenum,line1[k++],true,lmp));
    stateequationcount[i]++;
  }
  delete[] atomtypes;
}


void PairRANN::read_eos_constants(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,i1,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before state equations in potential file.");
  int n_body_type = nwords-4;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"equation of state element not found in atom types");}
  }
  i = atomtypes[0];
  found = false;
  for (k=0;k<stateequationperelement[i];k++) {
    if (state[i][k]->empty) {continue;}
    if (n_body_type!=state[i][k]->n_body_type) {continue;}
    for (j=0;j<n_body_type;j++) {
      if (state[i][k]->atomtypes[j]!=atomtypes[j]) {break;}
      if (j==n_body_type-1) {
        if (line[nwords-3].compare(state[i][k]->style)==0 && utils::inumeric(filename,linenum,line[nwords-2],true,lmp)==state[i][k]->id) {
          found=true;
          i1 = k;
          break;
        }
      }
    }
    if (found) {break;}
  }
  if (!found) {errorf(filename,linenum-1,"cannot define constants for unknown equation of state");}
  state[i][i1]->fullydefined=state[i][i1]->parse_values(line[nwords-1],line1);
  delete[] atomtypes;
}

void PairRANN::read_screening(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  if (nwords!=5)errorf(filename,linenum-1,"invalid screening command");
  int n_body_type = 3;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  j = atomtypes[1];
  k = atomtypes[2];
  int index = i*nelements*nelements+j*nelements+k;
  if (line[4].compare("Cmin")==0)  {
    screening_min[index] = utils::numeric(filename,linenum,line1[0],true,lmp);
  }
  else if (line[4].compare("Cmax")==0) {
    screening_max[index] = utils::numeric(filename,linenum,line1[0],true,lmp);
  }
  else errorf(filename,linenum-1,"unrecognized screening keyword");
  delete[] atomtypes;
}

//Called after finishing reading the potential file to make sure it is complete. True is bad.
//also allocates maxlayer and fingerprintlength and random weights and biases if ones were not provided.
bool PairRANN::check_potential(){
  int i,j,k,l;
  if (nelements==-1){errorf(FLERR,"elements not defined!");}
  for (i=0;i<=nelements;i++){
    if (i<nelements){
      if (mass[i]<0)errorf(FLERR,"mass not defined");//uninitialized mass
    }
    if (net[i].layers==0){
      continue;
    }//no definitions for this starting element, not considered an error.
    net[i].maxlayer=0;
    net[i].sumlayers=0;
    net[i].startI = new int [net[i].layers];
    for (j=0;j<net[i].layers;j++){
      net[i].startI[j] = net[i].sumlayers;
      net[i].sumlayers+=net[i].dimensions[j];
      if (net[i].dimensions[j]<=0)errorf(FLERR,"missing layersize");//incomplete network definition
      if (net[i].dimensions[j]>net[i].maxlayer)net[i].maxlayer = net[i].dimensions[j];
    }
    if (net[i].maxlayer>fnmax) {fnmax = net[i].maxlayer;}
    if (net[i].dimensions[net[i].layers-1]!=1)errorf(FLERR,"output layer must have single neuron");//output layer must have single neuron (the energy)
    if (net[i].dimensions[0]>fmax)fmax=net[i].dimensions[0];
    for (j=0;j<net[i].layers-1;j++){
      for (int k=0;k<net[i].dimensions[j+1];k++) {
        if (activation[i][j][k]->empty)errorf(FLERR,"undefined activations");//undefined activations
      }
      for (int k=0;k<net[i].bundles[j];k++){
        if (!bundle_inputdefined[i][j][k]){
          net[i].bundleinputsize[j][k] = net[i].dimensions[j];
          net[i].bundleinput[j][k] = new int[net[i].bundleinputsize[j][k]];
          for (int l=0;l<net[i].bundleinputsize[j][k];l++)net[i].bundleinput[j][k][l]=l;
        }
        if (!bundle_outputdefined[i][j][k]){
          net[i].bundleoutputsize[j][k] = net[i].dimensions[j+1];
          net[i].bundleoutput[j][k] = new int[net[i].bundleoutputsize[j][k]];
          for (int l=0;l<net[i].bundleoutputsize[j][k];l++)net[i].bundleoutput[j][k][l]=l;
        }
        if (net[i].identitybundle[j][k]){
          create_identity_wb(net[i].bundleinputsize[j][k],net[i].bundleoutputsize[j][k],i,j,k);
          if (net[i].bundleinputsize[j][k]!=net[i].bundleoutputsize[j][k]) errorf(FLERR,"input and output of identity bundles must be equal size");
        }
        if (!weightdefined[i][j][k] && !is_lammps)create_random_weights(net[i].bundleinputsize[j][k],net[i].bundleoutputsize[j][k],i,j,k);//undefined weights
        else if (!weightdefined[i][j][k]) errorf(FLERR,"undefined weight matrix!");
        if (!biasdefined[i][j][k] && !is_lammps)create_random_biases(net[i].dimensions[j+1],i,j,k);//undefined biases
        else if (!biasdefined[i][k][k]) errorf(FLERR,"undefined bias vector!");
      }
      for (int i1=0;i1<net[i].bundles[j];i1++){
        if (net[i].identitybundle[j][i1])continue;
      }
      for (int k=0;k<net[i].dimensions[j];k++){
        bool foundoutput=false;
        for (int l=0;l<net[i].bundles[j];l++){
          if (foundoutput)continue;
          for (int m=0;m<net[i].bundleinputsize[j][l];m++){
            if (net[i].bundleinput[j][l][m]==k){
              foundoutput = true;
              continue;
            }
          }
        }
        if (!foundoutput){
          errorf(FLERR,"found neuron with no output\n");}
      }
      for (int k=0;k<net[i].dimensions[j+1];k++){
        bool foundinput=false;
        for (int l=0;l<net[i].bundles[j];l++){
          if (foundinput)continue;
          for (int m=0;m<net[i].bundleoutputsize[j][l];m++){
            if (net[i].bundleoutput[j][l][m]==k){
              foundinput = true;
              continue;
            }
          }
        }
        if (!foundinput){errorf(FLERR,"found neuron with no input\n");}
      }
    }
    for (j=0;j<fingerprintperelement[i];j++){
      if (fingerprints[i][j]->fullydefined==false)errorf(FLERR,"undefined fingerprint parameters");
      fingerprints[i][j]->startingneuron = fingerprintlength[i];
      fingerprintlength[i] +=fingerprints[i][j]->get_length();
      if (fingerprints[i][j]->rc>cutmax){cutmax = fingerprints[i][j]->rc;}
      if (fingerprints[i][j]->spin==true){dospin=true;}
    }
    for (j=0;j<stateequationperelement[i];j++){
      if (state[i][j]->fullydefined==false)errorf(FLERR,"undefined state equation parameters");
      if (state[i][j]->rc>cutmax){cutmax = state[i][j]->rc;}
    }
    if (net[i].dimensions[0]!=fingerprintlength[i])errorf(FLERR,"fingerprint length does not match input layersize");
  }
  return false;//everything looks good
}

void PairRANN::create_identity_wb(int rows,int columns,int itype,int layer,int bundle){
  net[itype].bundleW[layer][bundle] = new double [rows*columns];
  net[itype].bundleB[layer][bundle] = new double [rows];
  net[itype].freezeW[layer][bundle] = new bool [rows*columns];
  net[itype].freezeB[layer][bundle] = new bool [rows];
  double r;
  for (int i=0;i<rows;i++){
    for (int j=0;j<columns;j++){
      net[itype].bundleW[layer][bundle][j*rows+i] = 0.0;
      net[itype].freezeW[layer][bundle][j*rows+i] = 1;
      if (i==j){
        net[itype].bundleW[layer][bundle][j*rows+i] = 1.0;
      }
    }
    net[itype].bundleB[layer][bundle][i] = 0.0;
    net[itype].freezeB[layer][bundle][i] = 0;
  }
  weightdefined[itype][layer][bundle]=true;
  biasdefined[itype][layer][bundle]=true;
}