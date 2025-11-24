CT=double(patient.CT);
sliceViewer(CT);

CTd=size(CT);
PTV=double(patient.SampledVoxels{1,1});
PTV1D=PTV(1,:)+PTV(2,:)*CTd(1)+PTV(3,:)*CTd(1)*CTd(2);

BLA=double(patient.SampledVoxels{1,2});
BLA1D=BLA(1,:)+BLA(2,:)*CTd(1)+BLA(3,:)*CTd(1)*CTd(2);

REC=double(patient.SampledVoxels{1,3});
REC1D=REC(1,:)+REC(2,:)*CTd(1)+REC(3,:)*CTd(1)*CTd(2);

CTmask=CT*0;
CT(PTV1D)=3000;
CT(BLA1D)=1000;
CT(REC1D)=2000;
figure
sliceViewer(CT,[]);

%%
CT=double(patient.CT);
Dij_PTV=double(data.matrix(1).A);
size_Dij_PTV=size(Dij_PTV);
x_vec=ones(size_Dij_PTV(2),1);
D_PTV=Dij_PTV*x_vec;
CT(PTV1D)=D_PTV*10000;

figure
sliceViewer(CT,[]);