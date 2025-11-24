CT=double(patient.CT);
% sliceViewer(CT);

CTd=size(CT);
PTV=double(patient.SampledVoxels{1,1});
PTV1D=PTV(1,:)+(PTV(2,:)-1)*CTd(1)+(PTV(3,:)-1)*CTd(1)*CTd(2);

BLA=double(patient.SampledVoxels{1,2});
BLA1D=BLA(1,:)+(BLA(2,:)-1)*CTd(1)+(BLA(3,:)-1)*CTd(1)*CTd(2);

REC=double(patient.SampledVoxels{1,3});
REC1D=REC(1,:)+(REC(2,:)-1)*CTd(1)+(REC(3,:)-1)*CTd(1)*CTd(2);

CTmask=CT*0;
CT(PTV1D)=3000;
CT(BLA1D)=1000;
CT(REC1D)=2000;
% figure
% sliceViewer(CT,[]);

%%
CT=double(patient.CT);
Dij_PTV=double(data.matrix(1).A);
size_Dij_PTV=size(Dij_PTV);
x_vec=rand(size_Dij_PTV(2),1);
D_PTV=Dij_PTV*x_vec;

Dij_BLA=double(data.matrix(5).A);
Dij_REC=double(data.matrix(3).A);

D_BLA=Dij_BLA*x_vec;
D_REC=Dij_REC*x_vec;

CT(PTV1D)=D_PTV*10000;
CT(BLA1D)=D_BLA*10000;
CT(REC1D)=D_REC*10000;

% figure
% sliceViewer(CT,[]);

%PTV
figure;
hist_DPTV=histogram(D_PTV, 100);
hist_Dx=hist_DPTV.BinEdges(1:end-1)+hist_DPTV.BinWidth/2;
hist_Dy=zeros(length(hist_Dx),1);
hist_total=sum(hist_DPTV.Values);
for i=1:length(hist_Dx)
    hist_Dy(length(hist_Dx)-i+1)=sum(hist_DPTV.Values(end-i+1:end))/hist_total*100;
end

%Bladder

figure;
hist_DBLA=histogram(D_BLA, 100);
hist_Bx=hist_DBLA.BinEdges(1:end-1)+hist_DBLA.BinWidth/2;
hist_By=zeros(length(hist_Bx),1);
hist_total=sum(hist_DBLA.Values);
for i=1:length(hist_Bx)
    hist_By(length(hist_Bx)-i+1)=sum(hist_DBLA.Values(end-i+1:end))/hist_total*100;
end

%rectum
figure;
hist_DREC=histogram(D_REC, 100);
hist_Rx=hist_DREC.BinEdges(1:end-1)+hist_DREC.BinWidth/2;
hist_Ry=zeros(length(hist_Dx),1);
hist_total=sum(hist_DREC.Values);
for i=1:length(hist_Rx)
    hist_Ry(length(hist_Rx)-i+1)=sum(hist_DREC.Values(end-i+1:end))/hist_total*100;
end

D95=interp1(hist_Dy(30:34),hist_Dx(30:34),95);
figure;
plot(hist_Dx/D95*100,hist_Dy);hold on
plot(hist_Bx/D95*100,hist_By);hold on
plot(hist_Rx/D95*100,hist_Ry);hold off
legend('PTV','BLA','REC')
xlabel('dose')
ylabel('volume')

%criteria
dose_cB=[80,75,70,65]/79.5*100; %BLA
vol_cB=[20,30,40,55];
dose_cR=[75,70,65,60]/79.5*100; %REC
vol_cR=[20,30,40,55];
vol_B=interp1(hist_Bx/D95*100,hist_By,dose_cB);
vol_R=interp1(hist_Rx/D95*100,hist_Ry,dose_cR);
score_B=sum(vol_cB>vol_B)
score_R=sum(vol_cR>vol_R)

dose_cP=87.12/79.5*100; %PTV
dose_P=mean(hist_Dx(98:100)/D95*100);
score_P=dose_cP>dose_P

score_all=score_P+score_B+score_R
