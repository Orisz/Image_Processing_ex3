%% Q1 - Frequancy Information - Section a
close all;clear all;clc;

uma = im2double(imread('..\Uma.JPG'));
uma_fft = fftshift(fft2(uma));

%ploting
figure()
subplot(1,2,1);
imshow(uma);
title('uma');
subplot(1,2,2);
imshow(log(1 + abs(uma_fft)),[]); %the [] changes the display range of a grayscale image (imshow uses [min(I(:)) max(I(:))])
title('uma after fft');

%% Section b - show low frequencies
[uma_size_x, uma_size_y] = size(uma_fft);

% choose 5% of lowest freq (2.5% to the left and to the right of the axis - because of the shift)
uma_fft_low_freq = zeros(uma_size_x, uma_size_y); %cancel other frequencies
x_min_freq = floor(uma_size_x/2-0.025*uma_size_x);
x_max_freq = uma_size_x - 1 - x_min_freq;
y_min_freq = floor(uma_size_y/2-0.025*uma_size_y);
y_max_freq = uma_size_y - 1 - y_min_freq;

% forming a new fft mat (contains only the lowest 5% frequencies)
uma_fft_low_freq(x_min_freq:x_max_freq , :) = uma_fft(x_min_freq:x_max_freq , :);
uma_fft_low_freq(: , y_min_freq:y_max_freq) = uma_fft(: , y_min_freq:y_max_freq);
uma_low_freq = ifft2(ifftshift(uma_fft_low_freq));

% plot
figure()
subplot(1,2,1);
imshow(abs(uma_low_freq));
title('uma - only low freq (5%) in each axis');
subplot(1,2,2);
imshow(log(1 + abs(uma_fft_low_freq)),[]);
title('uma fft - only low freq (5%) in each axis');

%% Section d - most dominant cols
rows_sum = sum(abs(uma_fft));
[rows_sum_sorted, position] = sort(rows_sum, 'descend');
cols_position = position(1 : (floor(0.05*length(position)))); % highest 5%
fprintf('most dominant colums are: ');
disp(cols_position)

%% Section e - most dominant rows
rows_sum = sum(abs(uma_fft), 2);
[rows_sum_sorted, position] = sort(rows_sum, 'descend');
rows_position = position(1 : (floor(0.05*length(position)))); % highest 5%
fprintf('most dominant rows are: ');
disp(rows_position')

%% Section f - show only dominant cols & rows
uma_dominant_fft = zeros(uma_size_x, uma_size_y);
uma_dominant_fft(: ,cols_position) = uma_fft(:, cols_position);
uma_dominant_fft(rows_position, :) = uma_fft(rows_position, :);
uma_dominant = ifft2(ifftshift(uma_dominant_fft));

figure()
subplot(1,2,1);
imshow(abs(uma_dominant));
title('uma dominant rows & cols');
subplot(1,2,2);
imshow(log(1 + abs(uma_dominant_fft)),[]);
title('uma dominant rows & cols fft');

%% Section g - most dominant frequencies
temp_mat = abs(uma_fft);
[Max, position] = sort(temp_mat(:), 'descend');
uma_position = position(1 : (floor(0.1*length(position)))); % highest 10%

%take only those frequencies
filter = zeros(512);
filter(uma_position) = 1;
uma_dominant_freq_fft = filter.*uma_fft;
uma_dominant_freq = ifft2(ifftshift(uma_dominant_freq_fft));

%plot
figure()
subplot(1,2,1);
imshow(abs(uma_dominant_freq));
title('uma 2D dominant frequencies');
subplot(1,2,2);
imshow(log(1 + abs(uma_dominant_freq_fft)),[]);
title('uma 2D dominant frequencies fft');

%% Q2 - Section a - show phase & amp of pictures
%regular photos % fft
cat = im2double(imread('..\cat.JPG'));
cat_fft = fft2(cat);
anna = im2double(imread('..\Anna.JPG'));
anna_fft = fft2(anna);

%get phase & amp
cat_fft_amp = abs(cat_fft);
cat_fft_phase = angle(cat_fft);
anna_fft_amp = abs(anna_fft);
anna_fft_phase = angle(anna_fft);

%ploting
figure()
subplot(2,2,2);
imshow(cat);
title('cat - regular photo');
subplot(2,2,1);
imshow(log(1 + abs(fftshift(cat_fft))),[]); %the [] changes the display range of a grayscale image (imshow uses [min(I(:)) max(I(:))])
title('cat after fft - amplitude');
subplot(2,2,4);
imshow(anna);
title('anna - regular photo');
subplot(2,2,3);
imshow(log(1 + abs(fftshift(anna_fft))),[]); 
title('anna after fft - amplitude');

%% Section b - show combined photos (swap phase and amp)
anna_amp_cat_phase_fft = anna_fft_amp.*exp(j*cat_fft_phase); %fourier transform
cat_amp_anna_phase_fft = cat_fft_amp.*exp(j*anna_fft_phase);

anna_amp_cat_phase = ifft2(anna_amp_cat_phase_fft);
cat_amp_anna_phase = ifft2(cat_amp_anna_phase_fft);

%ploting
figure()
subplot(1,2,1);
imshow(log(1 + abs(anna_amp_cat_phase)),[]);
title('mixed photo - amp of anna and phase of cat');
subplot(1,2,2);
imshow(log(1 + abs(cat_amp_anna_phase)),[]); 
title('mixed photo - phase of anna and amp of cat');

%% Section c - show combines photos (anna & random)
random_fft_amp = rand(size(anna_fft))*max(max(anna_fft_amp)); %rand returns numbers in interval (0,1)
random_fft_phase = rand(size(anna_fft))*2*pi;

anna_amp_random_phase_fft = anna_fft_amp.*exp(j*random_fft_phase); %fourier transform
random_amp_anna_phase_fft = random_fft_amp.*exp(j*anna_fft_phase);

anna_amp_random_phase = ifft2(anna_amp_random_phase_fft);
random_amp_anna_phase = ifft2(random_amp_anna_phase_fft);

%ploting
figure()
subplot(1,2,1);
imshow(log(1 + abs(anna_amp_random_phase)),[]);
title('mixed photo - amp of anna and phase of a random picture');
subplot(1,2,2);
imshow(log(1 + abs(random_amp_anna_phase)),[]); 
title('mixed photo - phase of anna and amp of a random picture');


%% Q 3 section b
clc;close all;clear all;
Tiffany = imread('../Tiffany.jpg');
Tiffany_DCTcoeff_8 = DCTcoeff(Tiffany,8);
Tiffany_DCTcoeff_64 = DCTcoeff(Tiffany,64);
Tiffany_DCTcoeff_512 = DCTcoeff(Tiffany,512);

%no logarithmics scailing
figure(1);
subplot(1,3,1);
imshow(mat2gray(Tiffany_DCTcoeff_8));
title('Tiffany DCTcoeff 8');
subplot(1,3,2);
imshow(mat2gray(Tiffany_DCTcoeff_64));
title('Tiffany DCTcoeff 64');
subplot(1,3,3);
imshow(mat2gray(Tiffany_DCTcoeff_512));
title('Tiffany DCTcoeff 512');
%with logarithmics scaling
figure(2);
subplot(1,3,1);
imshow(mat2gray(log10(1+abs(Tiffany_DCTcoeff_8))));
title('Tiffany DCTcoeff 8 log scale');
subplot(1,3,2);
imshow(mat2gray(log10(1+abs((Tiffany_DCTcoeff_64)))));
title('Tiffany DCTcoeff 64 log scale');
subplot(1,3,3);
imshow(mat2gray(log(1+abs((Tiffany_DCTcoeff_512)))));
title('Tiffany DCTcoeff 512 log scale');
%% Q 3 section e
L = 1/4;
[cols , rows] = size(Tiffany);
Tiffany_DCT_1 = DCT_L_coeff(Tiffany,1,L);%/4 as instructed
Tiffany_DCT_8 = DCT_L_coeff(Tiffany,8,L);
Tiffany_DCT_64 = DCT_L_coeff(Tiffany,64,L);
% recnstructed picture
Tiffany_reconst_1 = iDCTcoeff(Tiffany_DCT_1,1);
Tiffany_reconst_8 = iDCTcoeff(Tiffany_DCT_8,8);
Tiffany_reconst_64 = iDCTcoeff(Tiffany_DCT_64,64);
figure(3);
subplot(2,2,1);
imshow(Tiffany,[]);
title('Original Image');
subplot(2,2,2);
imshow(Tiffany_reconst_1,[]);
title('reconstructed Image from k = 1');
subplot(2,2,3);
imshow(Tiffany_reconst_8,[]);
title('reconstructed Image from k = 8');
subplot(2,2,4);
imshow(Tiffany_reconst_64,[]);
title('reconstructed Image from k = 64');

%%  Q 3 section f.1
absERR_1 = abs(im2double(Tiffany)-Tiffany_reconst_1);
absERR_8 = abs(im2double(Tiffany)-Tiffany_reconst_8);
absERR_64 = abs(im2double(Tiffany)-Tiffany_reconst_64);
figure(5);
subplot(1,3,1);
imshow(mat2gray(absERR_1));
colorbar;
title('Absolute Error K=1');
subplot(1,3,2);
imshow(mat2gray(absERR_8));
colorbar;
title('Absolute Error K=8')
subplot(1,3,3);
imshow(mat2gray(absERR_64));
colorbar;
title('Absolute Error K=64');
%%  Q 3 section f.2
K = [1,2,4,8,16,64,128,256];
MSE = [];
cur_Err = 0;
L = 1/4;
for i=1:length(K)
    
    dct_mat = DCT_L_coeff(Tiffany,K(i),L);
    reconstructed_pic = iDCTcoeff(dct_mat,K(i));
    cur_Err = immse(im2double(Tiffany) , reconstructed_pic);
    MSE = [MSE cur_Err];
end
figure(6);
stem(K , MSE);
grid on;
title('Mean Squared Error as a function of K');
xlabel('number of blocks per row/column');
ylabel('Mean Squared Error');

%% Q 4 part 1 section b
clc;clear all;close all;

[x,y]=meshgrid(0:1:255);
f1_x_y=sin(2*pi*(40/256)*y)+sin(2*pi*(5/256)*x)+sin(2*pi*(2/256)*(x+y));
f1_image=mat2gray(f1_x_y);
figure(1)
imshow(f1_image,[]);
title('f1 delta = 1','fontsize' , 20);

%% section c
F1_u_v = fftshift(fft2(f1_x_y));
F1_image = mat2gray(abs(F1_u_v));
figure(2);
imshow(F1_image);
title('F1(frequency domain) delta = 1','fontsize' , 20);
%% section d
[x5,y5]=meshgrid(0:5:255);
f5_x_y=sin(2*pi*(40/256)*y5)+sin(2*pi*(5/256)*x5)+sin(2*pi*(2/256)*(x5+y5));
f5_image=mat2gray(f5_x_y);
figure(3)
imshow(f5_image,[]);
title('f1 delta = 5','fontsize' , 20);

F5_u_v = fftshift(fft2(f5_x_y));
F5_image = mat2gray(abs(F5_u_v));
figure(4);
imshow(F5_image);
title('F5(frequency domain) delta = 1','fontsize' , 20);
%% section f

guss_filt = fspecial('gaussian',[7,7],5);
filtered_f1_x_y = imfilter(f1_x_y,guss_filt);

figure(5);
subplot (1,2,1);
imshow(mat2gray(filtered_f1_x_y));
title('Gaussian filtered picture','fontsize' , 20);

filtered_F1_u_v = fftshift(fft2(filtered_f1_x_y));
subplot(1,2,2);
filtered_F1_u_v_image = mat2gray(abs(filtered_F1_u_v));
imshow(filtered_F1_u_v_image);
title('Fourier transform of Gaussian filtered picture','fontsize' , 20);
%% section g
filtered_f5_x_y =  filtered_f1_x_y(1:5:end,1:5:end);

figure(6);
subplot (1,2,1);
imshow(mat2gray(filtered_f5_x_y));
title('Gaussian filtered picture downsampled by 5','fontsize' , 15);

filtered_F5_u_v = fftshift(fft2(filtered_f5_x_y));
filtered_F5_u_v_image = mat2gray(abs(filtered_F5_u_v));
subplot(1,2,2);
imshow(filtered_F5_u_v_image);
title('Fourier transform of Gaussian filtered picture downsampled by 5','fontsize' , 15);
%% section h + I
Baboon = imread('..\Baboon.jpg');
Baboon_gray = mat2gray(Baboon);
Baboon_fft=fftshift(fft2(Baboon_gray));

figure(7);
subplot(1,2,1);
imshow(Baboon_gray);
title('Baboon gray','fontsize' , 15);
fft_log=log10(1+abs(Baboon_fft));
subplot(1,2,2);
imshow(fft_log,[]);
title('Baboon FFT','fontsize' , 15);
%% section j
Baboon_downsamp = Baboon_gray(1:4:end,1:4:end);
Baboon_downsamp_fft=fftshift(fft2(Baboon_downsamp));

figure(8);
subplot(1,2,1);
imshow(Baboon_downsamp,[]);
title('Baboon downsampled by 4','fontsize' , 15);
fft_downsamp_log=log10(1+abs(Baboon_downsamp_fft));
subplot(1,2,2);
imshow(fft_downsamp_log,[]);
title('Baboon downsampled FFT','fontsize' , 15);
%% section k
guss_filt2 = fspecial('gaussian',[5,5],3);
filtered_Baboon = imfilter(Baboon_gray,guss_filt2);
filtered_Baboon_downsamp = filtered_Baboon(1:4:end,1:4:end);
FFT_filtered_Baboon_downsamp = fftshift(fft2(filtered_Baboon_downsamp));

figure(9);
subplot(1,2,1);
imshow(filtered_Baboon_downsamp,[]);
title('Baboon Gaussian filtered & downsampled by 4','fontsize' , 15);

subplot(1,2,2);
imshow(log( abs(FFT_filtered_Baboon_downsamp)+1),[]);
title('FFT Baboon filtered & downsampled by 4','fontsize' , 15);

%% Q 4 part b section a + b
clc;clear all;close all;
load ('..\SpongeBob.mat');
for i=2:1045
   if SpongeBob(:,:,i)== SpongeBob(:,:,1)
       break;
   end
end
round_trip = i;
vid=SpongeBob(:,:,1:round_trip);
implay(vid);
%% section d
SpongeBob_downsampled_18=SpongeBob(:,:,1:18:end);
implay(SpongeBob_downsampled_18);