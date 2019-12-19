function [C]=DCTcoeff( Pic,K )
[cols , rows] = size(Pic);
if(cols~=rows)
   throw('image should be square dimension');
end
M = cols / K;
Double_image = im2double(Pic);
%dct_coefs_mat = zeros(cols,rows);
for i=1:K
    for j=1:K
        Bloc_DCT = dct2(Double_image((i-1)*M+1:i*M,(j-1)*M+1:j*M));
        dct_coefs_mat((i-1)*M+1:i*M,(j-1)*M+1:j*M) = Bloc_DCT;
    end
end
C = dct_coefs_mat;
end

