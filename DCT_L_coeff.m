function [C]=DCT_L_coeff(Pic,K,L)
[cols , rows] = size(Pic);
if(cols~=rows)
   throw('image should be square dimension');
end
M = cols / K;
Double_image = im2double(Pic);
dct_coefs_mat = zeros(cols,rows);
%Bloc_DCT = zeros(M,M);
for i=1:K
    for j=1:K
        Bloc_DCT = dct2(Double_image((i-1)*M+1:i*M,(j-1)*M+1:j*M));
        [~,Indices] = sort(abs(Bloc_DCT(:)),'descend'); 
        tmpBloc = zeros(size(Bloc_DCT));
        tmpBloc(Indices(1:M*M*L)) = Bloc_DCT(Indices(1:M*M*L));
        dct_coefs_mat((i-1)*M+1:i*M,(j-1)*M+1:j*M) = tmpBloc;
    end
end
C = dct_coefs_mat;


end

