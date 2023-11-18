function pregunta3()

clc; clear;

numFolder=40;
numImg = 9;
S = [];

%Las variables T1, T2, T3... hacen referencia a variables temporales
% para realizar operaciones

for k=1:numFolder
  for i=1:numImg
  direccion=['training/s',num2str(k),'/',num2str(i),'.jpg'];
  T1=imread(direccion);
  T2=im2double(T1);
  T3=T2(:);
  S = [S,T3];
  endfor
endfor

  T4 = zeros(10304,1);
  for i=1:360
    T4 = T4 + S(:,i);
  endfor

  p = T4 / 360;

  A=[];

  for j=1:360
    T5= S(:,j) - p;
    A = [A,T5];
  endfor

  [Ur,Sr,Vr] = svdCompact(A);

  r = rank(A);

  ux = [];
  for k=1:r
      T6 = Ur(:,k);
      ux = [ux, T6];
    endfor
%SC es analoga a la matriz S pero con las imagenes del compare
  SC = [];
  numImgCompare = 40;
  for i=1:numImgCompare
  direccion2=['compare/p',num2str(i),'.jpg'];
  V1=imread(direccion2);
  V2=im2double(V1);
  V3=V2(:);
  SC = [SC,V3];
endfor

%Ei es el epsilon en i
%ME es la matriz de Epsilons

for i=1:40
  x = ux' * (SC(:,i)-p);
  ME=[];
  for j=1:360
    xi = ux' * (S(:,j) - p);
    Ei = norm(x-xi);

    ME = [ME,Ei];
  endfor

  [Xmin,Ymin]= min(ME);

  VecR = S(:,Ymin);

  Img1 = reshape(SC(:,i),[112,92]);
  Img2 = reshape(VecR, [112,92]);

  subplot(1,2,1);
    imshow(Img1,[]);
    subplot(1,2,2);
    imshow(Img2,[]);
    pause(2);
  endfor
  endfunction
