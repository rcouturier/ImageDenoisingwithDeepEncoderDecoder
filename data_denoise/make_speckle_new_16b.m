
out='path/pix2pix_val_speckle16_big_div4_truncated/';
in1='path/cover/';



for i=1800:2000
%for i=union(1:1799,2001:3000)

		name=strcat(in1,num2str(i),'.pgm');
		im1=imread(name);
		im1=uint16(im1);

		im2=addSpeckleNoise(name,1,16,1);

		im2=im2/4;
		im2(im2>256)=256;

		size(im2)
		im=double(cat(2,im1,im2))/256;


		imwrite(im,strcat(out,num2str(i),'.png'));


end
