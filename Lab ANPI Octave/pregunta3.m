function X=pregunta3()

clc; clear;

numFolder=40;
numImg = 9;
S = []

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
    T4 = T4 + S(1,i);
  endfor

  p = X / 360;

  A=[];

  for j=1:360
    T5= S(j) - p;
    A = [A,T5];
  endfor

  [Ur,Sr,Vr] = svdCompact(A);

  endfunction
