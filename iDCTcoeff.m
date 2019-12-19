function [Pic]=iDCTcoeff(C,K)
[cols , rows] = size(C);
if(cols~=rows)
   throw('image should be square dimension');
end
M = cols / K;
Double_image = im2double(C);
Pic_mat = zeros(cols,rows);
for i=1:K
    for j=1:K
        Bloc_Pic = idct2(Double_image((i-1)*M+1:i*M,(j-1)*M+1:j*M));
        Pic_mat((i-1)*M+1:i*M,(j-1)*M+1:j*M) = Bloc_Pic;
    end
end
Pic = Pic_mat;

end

