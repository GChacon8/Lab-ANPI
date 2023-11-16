function X=pregunta1()
  tiempos1 = [];
  tiempos2 = [];
  for i=2:3%5:12
    A=rand(2^i, 2^(i-1));
    tic
    [U,S,V] = svd(A);
    display(U);
    display(S);
    display(V);
    tiempos1 = [tiempos1, toc];
    tic
    [Ur,Sr,Vr] = svdCompact(A);
    display(Ur);
    display(Sr);
    display(Vr);
    tiempos2 = [tiempos2, toc];
  endfor

  x = [5,6,7,8,9,10,11,12];
  plot(x, tiempos1, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'SVD Octave');
  hold on;
  plot(x, tiempos2, 'LineWidth', 2 , 'Color', 'blue', 'DisplayName', 'SVD Compact');
  title("Gráfico SVD vs SVD-Compact");
  xlabel("Parametro k");
  ylabel("Tiempo");
  legend;
   grid on;
  endfunction
