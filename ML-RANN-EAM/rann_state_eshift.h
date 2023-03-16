
#ifndef LMP_RANN_STATE_ESHIFT_H
#define LMP_RANN_STATE_ESHIFT_H

#include "rann_stateequation.h"
#include "pair_rann.h"

namespace LAMMPS_NS {
namespace RANN {

  class State_eshift : public State {
   public:
    State_eshift(class PairRANN *);
    ~State_eshift();
    void eos_function(double*,double**,int,int,double*,double*,double*,int*,int,int*);
    bool parse_values(std::string, std::vector<std::string>);
    void write_values(FILE *); 
    void init(int *,int);
    double eshift;
  };

    State_eshift::State_eshift(PairRANN *_pair) : State(_pair) {
        eshift = 0;
        n_body_type = 1;
        style = "eshift";
        atomtypes = new int[n_body_type];
        empty = true;
        fullydefined = false;
        _pair->allscreen = false;
    }

State_eshift::~State_eshift()
{
  delete [] atomtypes;
}

//called after state equation is declared for i-j type, but before its parameters are read.
void State_eshift::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_eshift::eos_function(double *ep,double **force,int ii,int nn,
                                double *xn,double *yn,double *zn,int *tn,int jnum,int* jl)
{
  ep[0]+=eshift;
  return;
}

bool State_eshift::parse_values(std::string constant,std::vector<std::string> line1) {
  int nwords=line1.size();
  if (constant.compare("eshift")==0) {
    eshift = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for eshift equation of state");
 //allow undefined delta, (default = 0)
  return true;
}

void State_eshift::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:eshift:\n",style,id);
  fprintf(fid,"%f\n",eshift);
}


}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* LMP_RANN_STATE_ROSE_H_ */
