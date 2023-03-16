#include "pair_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"

using namespace LAMMPS_NS;

//energy and error per simulation
void PairRANN::write_debug_level1(double *fit_err,double *val_err) {
	int check = mkdir("DEBUG",0777);
	char debug_summary [strlen(potential_output_file)+14];
	sprintf(debug_summary,"DEBUG/%s.debug1",potential_output_file);
	FILE *fid_summary = fopen(debug_summary,"w");
	fprintf(fid_summary,"#Error from simulations included in training:\n");
	fprintf(fid_summary,"#training_size, validation_size:\n");
	fprintf(fid_summary,"%d %d\n",nsimr,nsimv);
	fprintf(fid_summary,"#fit_index, filename, timestep, natoms, target/atom, state/atom, (t-s)/a, error/atom\n");
	double targetmin = 10^300;
	double errmax = 0;
	double err = 0;
	double target = 0;
	for (int i=0;i<nsimr;i++){
		err = fit_err[i]/sims[r[i]].energy_weight/sims[r[i]].inum;
		target = sims[r[i]].energy/sims[r[i]].inum;
		double state = sims[r[i]].state_e/sims[r[i]].inum;
		if (target<targetmin)targetmin=target;
		if (fabs(err)>errmax)errmax=abs(err);
		fprintf(fid_summary,"%d %s %d %d %f %f %f %f\n",r[i],sims[r[i]].filename,sims[r[i]].timestep,sims[r[i]].inum,target+state,state,target,err);
	}
	fprintf(fid_summary,"#Error from simulations included in validation:\n");
	fprintf(fid_summary,"#val_index, filename, timestep, natoms, target/atom, state/atom, (t-s)/a, error/atom\n");
	for (int i=0;i<nsimv;i++){
		err = val_err[i]/sims[v[i]].energy_weight/sims[v[i]].inum;
		target = sims[v[i]].energy/sims[v[i]].inum;
		double state = sims[v[i]].state_e/sims[v[i]].inum;
		if (target<targetmin)targetmin=target;
		if (fabs(err)>errmax)errmax=abs(err);
		fprintf(fid_summary,"%d %s %d %d %f %f %f %f\n",v[i],sims[v[i]].filename,sims[v[i]].timestep,sims[v[i]].inum,target+state,state,target,err);
	}
	fprintf(fid_summary,"#mininum target/atom: %f, max error/atom: %f\n",targetmin,errmax);
	fclose(fid_summary);
}

//dump files with energy of each atom
void PairRANN::write_debug_level2(double *fit_err,double *val_err) {
 FILE *dumps[nsets];
 FILE *current;
 double **energies;
 int index[nsims];
 for (int i = 0;i<nsims;i++){
	 index[i]=i;
 }
 energies = new double*[nsims];
 get_per_atom_energy(energies,index,nsims,net);
 char *debugnames[nsets];
 int check = mkdir("DEBUG",0777);
 for (int i=0;i<nsets;i++){
	 if (Xset[i]==0){continue;}
	 debugnames[i] = new char [strlen(dumpfilenames[i])+20];
	 sprintf(debugnames[i],"DEBUG/%s.debug2",dumpfilenames[i]);
	 dumps[i]=fopen(debugnames[i],"w");
 }
 for (int i=0;i<nsims;i++){
	 bool foundcurrent=false;
	 int nsims,j;
	 for (j=0;j<nsets;j++){
		 if (Xset[j]==0){continue;}
		 if (strcmp(dumpfilenames[j],sims[i].filename)==0){
			 current = dumps[j];
			 foundcurrent = true;
			 nsims = Xset[j];
			 break;
		 }
	 }
	 if (!foundcurrent){errorf("something happened!\n");}
	 fprintf(current,"ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
	 fprintf(current,"%d %f %f %f %d %f %f %f %f %f %f\n",sims[i].timestep,sims[i].energy,sims[i].energy_weight,sims[i].force_weight,nsims,sims[i].spinvec[0],sims[i].spinvec[1],sims[i].spinvec[2],sims[i].spinaxis[0],sims[i].spinaxis[1],sims[i].spinaxis[2]);
	 fprintf(current,"ITEM: NUMBER OF ATOMS\n");
	 fprintf(current,"%d\n",sims[i].inum);
	 fprintf(current,"ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
	 double xmin = sims[i].origin[0];
	 double xmax = sims[i].box[0][0]+sims[i].origin[0];
	 double ymin = sims[i].origin[1];
	 double ymax = sims[i].box[1][1]+sims[i].origin[1];
	 double zmin = sims[i].origin[2];
	 double zmax = sims[i].box[2][2]+sims[i].origin[2];
	 if (sims[i].box[0][1]>0){xmax += sims[i].box[0][1];}
	 else {xmin += sims[i].box[0][1];}
	 if (sims[i].box[0][2]>0){xmax += sims[i].box[0][2];}
	 else {xmin += sims[i].box[0][2];}
	 if (sims[i].box[1][2]>0){ymax += sims[i].box[1][2];}
	 else {ymax += sims[i].box[1][2];}
	 fprintf(current,"%f %f %f\n",xmin,xmax,sims[i].box[0][1]);
	 fprintf(current,"%f %f %f\n",ymin,ymax,sims[i].box[0][2]);
	 fprintf(current,"%f %f %f\n",zmin,zmax,sims[i].box[1][2]);
	 fprintf(current,"ITEM: ATOMS id type x y z c_eng\n");
	 for (int j=0;j<sims[i].inum;j++){
		 fprintf(current,"%d %d %f %f %f %f\n",sims[i].ilist[j],sims[i].type[j],sims[i].x[j][0],sims[i].x[j][1],sims[i].x[j][2],energies[i][j]+sims[i].state_ea[j]);
	 }
 }
 for (int i=0;i<nsets;i++){
	 if (Xset[i]==0){continue;}
	 fclose(dumps[i]);
 }
}

//calibration data: current Jacobian, target vector, solve vector, and step.
void PairRANN::write_debug_level3(double *jacob,double *target,double *beta,double *delta) {
	int check = mkdir("DEBUG",0777);
	FILE *fid = fopen("DEBUG/jacobian.debug3.csv","w");
	for (int i =0;i<jlen1;i++){
		for (int j=0;j<betalen;j++){
			fprintf(fid,"%f, ",jacob[i*betalen+j]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
	fid = fopen("DEBUG/target.debug3.csv","w");
	for (int i=0;i<jlen1;i++){
		fprintf(fid,"%f\n",target[i]);
	}
	fclose(fid);
	fid = fopen("DEBUG/beta.debug3.csv","w");
	for (int j=0;j<betalen;j++){
		fprintf(fid,"%f\n",beta[j]);
	}
	fclose(fid);
	fid = fopen("DEBUG/delta.debug3.csv","w");
	for (int j=0;j<betalen;j++){
		fprintf(fid,"%f\n",delta[j]);
	}
	fclose(fid);
}

//dump files with energy and forces of each atom
//SLOW - recomputes fingerprints because memory storage is unreasonable to keep everything from first time.
void PairRANN::write_debug_level4(double *fit_err,double *val_err) {
	printf("starting debug level 4\n");
	//double est=0;
	//for (int nn=0;nn<nsims;nn++){
	//	est+=sims[nn].time;
	//}
	//printf("estimated time: %f seconds\n",est/omp_get_num_threads());
FILE *dumps[nsims];
FILE *current;
char *debugnames[nsims];
 int check = mkdir("DEBUG",0777);
 for (int i=0;i<nsims;i++){
	 debugnames[i] = new char [strlen(sims[i].filename)+20];
	 sprintf(debugnames[i],"DEBUG/%s.debug4.%d",sims[i].filename,sims[i].timestep);
	 dumps[i]=fopen(debugnames[i],"w");
 }
#pragma omp parallel
{
int i,ii,itype,f,jnum,len,j,nn;
#pragma omp for schedule(guided)
for (nn=0;nn<nsims;nn++){
	current = dumps[nn];
	 int j;
	double *force[sims[nn].inum+sims[nn].gnum];
	double force1[(sims[nn].inum+sims[nn].gnum)*3];
	double *fm[sims[nn].inum+sims[nn].gnum];
	double fm1[(sims[nn].inum+sims[nn].gnum)*3];
	for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
		force[ii]=&force1[ii*3];
		fm[ii]=&fm1[ii*3];
	}
	 double energy[sims[nn].inum];
	 for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
		for (j=0;j<3;j++){
			force[ii][j]=0;
			fm[ii][j]=0;
		}
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
		//TO D0: add check against stack size
		double dSijkx[jnum*jnum];
		double dSijky[jnum*jnum];
		double dSijkz[jnum*jnum];
		bool Bij[jnum];
		double sx[jnum*f];
		double sy[jnum*f];
		double sz[jnum*f];
		for (j=0;j<f*jnum;j++){
			sx[j]=sy[j]=sz[j]=0;
		}
		if (doscreen){
			screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
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
		double e=0.0;
		double e1 = 0.0;
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
		itype = map[sims[nn].type[i]];
		propagateforward(&e1,force,ii,jnum,itype,features,dfeaturesx,dfeaturesy,dfeaturesz,jl,nn); 
		energy[ii] = e+e1;
	}
	//apply ghost neighbor forces back into box
	for (ii = sims[nn].inum;ii<sims[nn].inum+sims[nn].gnum;ii++){
		for (j=0;j<3;j++){
			force[sims[nn].id[ii]][j]+=force[ii][j];
		}
	}
	i=nn;
	 fprintf(current,"ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
	 fprintf(current,"%d %f %f %f %d %f %f %f %f %f %f\n",sims[i].timestep,sims[i].energy,sims[i].energy_weight,sims[i].force_weight,nsims,sims[i].spinvec[0],sims[i].spinvec[1],sims[i].spinvec[2],sims[i].spinaxis[0],sims[i].spinaxis[1],sims[i].spinaxis[2]);
	 fprintf(current,"ITEM: NUMBER OF ATOMS\n");
	 fprintf(current,"%d\n",sims[i].inum);
	 fprintf(current,"ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
	 double xmin = sims[i].origin[0];
	 double xmax = sims[i].box[0][0]+sims[i].origin[0];
	 double ymin = sims[i].origin[1];
	 double ymax = sims[i].box[1][1]+sims[i].origin[1];
	 double zmin = sims[i].origin[2];
	 double zmax = sims[i].box[2][2]+sims[i].origin[2];
	 if (sims[i].box[0][1]>0){xmax += sims[i].box[0][1];}
	 else {xmin += sims[i].box[0][1];}
	 if (sims[i].box[0][2]>0){xmax += sims[i].box[0][2];}
	 else {xmin += sims[i].box[0][2];}
	 if (sims[i].box[1][2]>0){ymax += sims[i].box[1][2];}
	 else {ymax += sims[i].box[1][2];}
	 fprintf(current,"%f %f %f\n",xmin,xmax,sims[i].box[0][1]);
	 fprintf(current,"%f %f %f\n",ymin,ymax,sims[i].box[0][2]);
	 fprintf(current,"%f %f %f\n",zmin,zmax,sims[i].box[1][2]);
	 fprintf(current,"ITEM: ATOMS id type x y z c_eng fx fy fz\n");
	 for (int j=0;j<sims[i].inum;j++){
		 fprintf(current,"%d %d %f %f %f %f %f %f %f\n",sims[i].ilist[j],sims[i].type[j],sims[i].x[j][0],sims[i].x[j][1],sims[i].x[j][2],energy[j],force[j][0],force[j][1],force[j][2]);
	 }
	 fclose(dumps[i]);
}
}
}

//dump files with numerical forces on each atom along with analytical forces
//VERY SLOW - computes fingerprints for each simulation 6*inum times.
void PairRANN::write_debug_level5(double *fit_err,double *val_err) {
printf("starting debug level 5\n");
double diff = 1e-6;
//double est=0;
//for (int nn=0;nn<nsims;nn++){
//	est+=sims[nn].time;
//}
//printf("estimated time: %f seconds\n",est/omp_get_num_threads());
FILE *dumps[nsims];
FILE *current;
char *debugnames[nsims];
int check = mkdir("DEBUG",0777);
for (int i=0;i<nsims;i++){
	debugnames[i] = new char [strlen(sims[i].filename)+20];
	sprintf(debugnames[i],"DEBUG/%s.debug5.%d",sims[i].filename,sims[i].timestep);
	dumps[i]=fopen(debugnames[i],"w");
}
#pragma omp parallel
{
int i,ii,itype,f,jnum,len,j,nn,s,k,v;
#pragma omp for schedule(guided)
for (nn=0;nn<nsims;nn++){
	 int j;
	 current = dumps[nn];
	 double energy1[sims[nn].inum][3][2];
	 double e1[sims[nn].inum];
	 double fn[sims[nn].inum][3];
	 double fa[sims[nn].inum][3];
	 for (k=0;k<sims[nn].inum;k++){
		 for (v=0;v<3;v++){
		 	for (s=-1;s<2;s=s+2){
				 sims[nn].x[k][v]+=s*diff;
				 for (ii=sims[nn].inum;ii<sims[nn].inum+sims[nn].gnum;ii++){
					 if (sims[nn].id[ii]==k)sims[nn].x[ii][v]+=s*diff;
				 }
				double *force[sims[nn].inum+sims[nn].gnum];
				double force1[(sims[nn].inum+sims[nn].gnum)*3];
				double *fm[sims[nn].inum+sims[nn].gnum];
				double fm1[(sims[nn].inum+sims[nn].gnum)*3];
				for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
					force[ii]=&force1[ii*3];
					fm[ii]=&fm1[ii*3];
				}
				double energy[sims[nn].inum];
				for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
					for (j=0;j<3;j++){
						force[ii][j]=0;
						fm[ii][j]=0;
					}
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
					//TO D0: add check against stack size
					double dSijkx[jnum*jnum];
					double dSijky[jnum*jnum];
					double dSijkz[jnum*jnum];
					bool Bij[jnum];
					double sx[jnum*f];
					double sy[jnum*f];
					double sz[jnum*f];
					for (j=0;j<f*jnum;j++){
						sx[j]=sy[j]=sz[j]=0;
					}
					if (doscreen){
						screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
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
					double e=0.0;
					double e1 = 0.0;
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
					itype = map[sims[nn].type[i]];
					propagateforward(&e1,force,ii,jnum,itype,features,dfeaturesx,dfeaturesy,dfeaturesz,jl,nn); 
					energy[ii] = e+e1;
				}
				//apply ghost neighbor forces back into box
				for (ii = sims[nn].inum;ii<sims[nn].inum+sims[nn].gnum;ii++){
					for (j=0;j<3;j++){
						force[sims[nn].id[ii]][j]+=force[ii][j];
					}
				}
				sims[nn].x[k][v]-=s*diff;
				for (ii=sims[nn].inum;ii<sims[nn].inum+sims[nn].gnum;ii++){
				  if (sims[nn].id[ii]==k)sims[nn].x[ii][v]-=s*diff;
				}
				energy1[k][v][(s+1)>>1]=0;
				for (ii=0;ii<sims[nn].inum;ii++)energy1[k][v][(s+1)>>1]+=energy[ii];
			}
			fn[k][v]=(energy1[k][v][0]-energy1[k][v][1])/diff/2;
		 }
	 }
	double *force[sims[nn].inum+sims[nn].gnum];
	double force1[(sims[nn].inum+sims[nn].gnum)*3];
	double *fm[sims[nn].inum+sims[nn].gnum];
	double fm1[(sims[nn].inum+sims[nn].gnum)*3];
	for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
		force[ii]=&force1[ii*3];
		fm[ii]=&fm1[ii*3];
	}
	double energy[sims[nn].inum];
	for (ii=0;ii<sims[nn].inum+sims[nn].gnum;ii++){
		for (j=0;j<3;j++){
			force[ii][j]=0;
			fm[ii][j]=0;
		}
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
		//TO D0: add check against stack size
		double dSijkx[jnum*jnum];
		double dSijky[jnum*jnum];
		double dSijkz[jnum*jnum];
		bool Bij[jnum];
		double sx[jnum*f];
		double sy[jnum*f];
		double sz[jnum*f];
		for (j=0;j<f*jnum;j++){
			sx[j]=sy[j]=sz[j]=0;
		}
		if (doscreen){
			screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
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
		double e=0.0;
		double e1 = 0.0;
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
		itype = map[sims[nn].type[i]];
		propagateforward(&e1,force,ii,jnum,itype,features,dfeaturesx,dfeaturesy,dfeaturesz,jl,nn); 
		energy[ii] = e+e1;
	}
	//apply ghost neighbor forces back into box
	for (ii = sims[nn].inum;ii<sims[nn].inum+sims[nn].gnum;ii++){
		for (j=0;j<3;j++){
			force[sims[nn].id[ii]][j]+=force[ii][j];
		}
	}
	i=nn;
	 fprintf(current,"ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
	 fprintf(current,"%d %f %f %f %d %f %f %f %f %f %f\n",sims[i].timestep,sims[i].energy,sims[i].energy_weight,sims[i].force_weight,nsims,sims[i].spinvec[0],sims[i].spinvec[1],sims[i].spinvec[2],sims[i].spinaxis[0],sims[i].spinaxis[1],sims[i].spinaxis[2]);
	 fprintf(current,"ITEM: NUMBER OF ATOMS\n");
	 fprintf(current,"%d\n",sims[i].inum);
	 fprintf(current,"ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
	 double xmin = sims[i].origin[0];
	 double xmax = sims[i].box[0][0]+sims[i].origin[0];
	 double ymin = sims[i].origin[1];
	 double ymax = sims[i].box[1][1]+sims[i].origin[1];
	 double zmin = sims[i].origin[2];
	 double zmax = sims[i].box[2][2]+sims[i].origin[2];
	 if (sims[i].box[0][1]>0){xmax += sims[i].box[0][1];}
	 else {xmin += sims[i].box[0][1];}
	 if (sims[i].box[0][2]>0){xmax += sims[i].box[0][2];}
	 else {xmin += sims[i].box[0][2];}
	 if (sims[i].box[1][2]>0){ymax += sims[i].box[1][2];}
	 else {ymax += sims[i].box[1][2];}
	 fprintf(current,"%f %f %f\n",xmin,xmax,sims[i].box[0][1]);
	 fprintf(current,"%f %f %f\n",ymin,ymax,sims[i].box[0][2]);
	 fprintf(current,"%f %f %f\n",zmin,zmax,sims[i].box[1][2]);
	 fprintf(current,"ITEM: ATOMS id type x y z c_eng fax fnx fay fny faz fnz\n");
	 for (int j=0;j<sims[i].inum;j++){
		 fprintf(current,"%d %d %f %f %f %f %f %f %f %f %f %f\n",sims[i].ilist[j],sims[i].type[j],sims[i].x[j][0],sims[i].x[j][1],sims[i].x[j][2],energy[j],force[j][0],fn[j][0],force[j][1],fn[j][1],force[j][2],fn[j][2]);
	 }
	 fclose(dumps[i]);
}
}

}

//Calculate numerical jacobian and difference between numerical and analytical jacobian
void PairRANN::write_debug_level6(double *fit_err,double *val_err) {
	printf("starting debug level 6\n");
	double diff = 1e-6;

}