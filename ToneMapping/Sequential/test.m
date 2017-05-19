% files = {'test2/02_Dec.jpg', 'test2/01_Dec.jpg', 'test2/original.jpg', 'test2/01_Inc.jpg', 'test2/02_Inc.jpg'};
% expTimes = [0.25, 0.5, 1, 1.5, 2];
% 
% hdr = makehdr(files, 'RelativeExposure', expTimes ./ expTimes(1));
% 
% hdrwrite(hdr, 'test2/result.hdr');

fileID = fopen('times.csv','w');
fprintf(fileID, 'image,dimensions,time\n');

for i={'testsHDR/test1.hdr', 'testsHDR/test2.hdr', 'testsHDR/test3.hdr', 'testsHDR/test4.hdr', 'testsHDR/test5.hdr'}
    hdr = hdrread(i{1});
    [width, height, channels] = size(hdr);
    image_size = strcat(num2str(width), 'x', num2str(height));
    f = @() tonemap(hdr);
    fprintf(fileID, '%s,%s,%f\n', i{1}, image_size, timeit(f));
end

fclose(fileID);

% rgb = tonemap(hdr);
%  figure
%  imshow(f)