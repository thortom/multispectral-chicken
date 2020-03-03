path(path, '/home/thor/HI/Lokaverkefni/Code/Scripts/ics_reader')
path(path, '/home/thor/MATLAB Add-Ons/Apps/Slicer')
path(path, '/home/thor/HI/PreviousCourses/RAF512M/Lokaverkefni/Slicer/matImage')
installMatImage

SlicerApp % Then load in the variable

%%
icsfile = '/media/thor/My Passport/Marel/TomraFiles/IC-Fish_copyFromSindri/fx17/2019-01-23_09-59-30_fx17-18-scan1-XRT-00022.ICS';
X17 = readICSFile(icsfile);

imageFX17 = X17.data;
imageFX17 = permute(imageFX17,[2 3 1]);
imagesc(imageFX17(:, :, 50)); colormap(gray);

%%
icsfile = '/media/thor/My Passport/Marel/TomraFiles/IC-Fish_copyFromSindri/fx10/2019-01-24_14-09-20_fx10-85-scan1-XRT-00080.ICS';
X10 = readICSFile(icsfile);

imageFX10 = X10.data;
imageFX10 = permute(imageFX10,[2 3 1]);
imagesc(imageFX10(:, :, 150)); colormap(gray);

%%
icsfile = '/media/thor/My Passport/Marel/TomraFiles/IC-Fish_copyFromSindri/laser/2019-01-23_16-39-49_laser-48-VIS-00052.ICS';
X = readICSFile(icsfile);

imageLaser = X.data;
imageLaser = permute(imageLaser,[2 3 1]);
imagesc(imageLaser); colormap(gray);

%%
icsfile = '/media/thor/My Passport/Marel/TomraFiles/IC-Fish_copyFromSindri/rgb/2019-01-24_16-44-26_rgb-220-VIS-00164.ICS';
X = readICSFile(icsfile);

imageRGB = X.data;
imageRGB = permute(imageRGB,[2 3 1]);
imagesc(imageRGB); colormap(gray);

%%
readname = 'chicken_fm.tif';
tiff_stack = imread(readname, 1) ; % read in first image

%tiff_stack = tiff_stack(xmin:xmax , ymin:ymax);
tiff_allBands = mat2gray(tiff_stack);

%concatenate each successive tiff to tiff_stackimageLabeler
for ii = 2 : 7
    temp_tiff = imread(readname, ii);
    tiff_stack = cat(3 , tiff_stack, temp_tiff);
end

imageLabeler(tiff_stack)

% https://labelbox.com/blog/introducing-image-segmentation/
% https://ch.mathworks.com/help/vision/ug/label-pixels-for-semantic-segmentation.html

%% The main processing loop. That loops over all files in folder and saves them as mat images
myDir = '/home/thor/HI/Lokaverkefni/Code/data/TomraData/day1/legmeat'; %gets directory
myFiles = dir(fullfile(myDir,'*.ics')); %gets all ics files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);

    X = readICSFile(fullFileName);

    figure()
    image = X.data;
    image = permute(image,[2 3 1]);
    imagesc(image(:, :, 150)); colormap(gray);

    fileName = strrep(fullFileName,'.ics','.mat');
    save(fileName, '-double', 'image');
end
