%
% Corrupts an image with speckle noise and outputs the noisy image
%
% Params :
% - image_name (string) : filepath of the image to be corrupted
% - L : order of the gamma function used to draw the speckle values.
% - res : resolution, in bits, of the output image (8 or 16)
% - mae : pre-scaling coefficient applied to minimize gray-level
%   saturation after speckle corruption (depends on the optical
%   parameters). This mae parameter is used only for 8-bit output image; As for 16-bit
%   output, dynamic is considered high enough to avoid pre-scaling and this
%   parameter can be ommited.
%
function Inoisy = addSpeckleNoise(file, L, res, mae)
    Icov = double(imread(file));
    imgSize = size(Icov);
    
    if res==8
        Icov = Icov / mae ;
        Inoisy = uint8(Icov .* gamma_rand_ordre_entier(L, 1, imgSize(1), imgSize(2))) ;
        
    else
        Inoisy = uint16(Icov .* gamma_rand_ordre_entier(L, 1, imgSize(1), imgSize(2))) ;
    end
end